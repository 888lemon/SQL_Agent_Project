import json
from pathlib import Path

from sqlglot import exp, parse_one

from app.core.retriever import SchemaRetriever
from app.db.handler import SQLiteHandler
from app.core.security import SQLSecurityAudit


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_predictions(predictions_path):
    if not predictions_path:
        return {}

    path = Path(predictions_path)
    with open(path, "r", encoding="utf-8") as file:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in file if line.strip()]
        else:
            rows = json.load(file)

    if isinstance(rows, dict):
        return rows
    return {row["id"]: row for row in rows}


def build_retriever(db_path, metadata_path, model_name="all-MiniLM-L6-v2"):
    db = SQLiteHandler(db_path)
    retriever = SchemaRetriever(db_handler=db, model_name=model_name)
    retriever.build_index(metadata_path)
    return retriever


def build_db_handler(db_path):
    return SQLiteHandler(db_path)


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _normalize_columns(columns):
    return {(item["table"], item["column"]) for item in columns}


def _normalize_scalar(value):
    if isinstance(value, float):
        return round(value, 6)
    if value is None:
        return None
    return str(value)


def normalize_result_rows(result):
    rows = result.get("data", []) if isinstance(result, dict) else result
    normalized = []
    for row in rows:
        normalized.append(tuple(_normalize_scalar(cell) for cell in row))
    normalized.sort()
    return normalized


def evaluate_retrieval_case(retriever, case):
    result = retriever.get_relevant_schema(case["question"])

    predicted_tables = set(result.get("selected_tables", []))
    expected_tables = set(case.get("expected_tables", []))

    predicted_columns = set(result.get("selected_columns", []))
    expected_columns = _normalize_columns(case.get("expected_columns", []))

    predicted_metrics = set(result.get("selected_metrics", []))
    expected_metrics = set(case.get("expected_metrics", []))

    predicted_joins = set(result.get("selected_joins", []))
    expected_joins = set(case.get("expected_joins", []))

    return {
        "id": case["id"],
        "question": case["question"],
        "retrieval": {
            "predicted_tables": sorted(predicted_tables),
            "missing_tables": sorted(expected_tables - predicted_tables),
            "extra_tables": sorted(predicted_tables - expected_tables),
            "table_recall": _safe_ratio(len(expected_tables & predicted_tables), len(expected_tables)),
            "table_exact_match": predicted_tables == expected_tables,
            "predicted_columns": sorted([{"table": table, "column": column} for table, column in predicted_columns], key=lambda item: (item["table"], item["column"])),
            "column_recall": _safe_ratio(len(expected_columns & predicted_columns), len(expected_columns)),
            "predicted_metrics": sorted(predicted_metrics),
            "metric_recall": _safe_ratio(len(expected_metrics & predicted_metrics), len(expected_metrics)),
            "predicted_joins": sorted(predicted_joins),
            "join_recall": _safe_ratio(len(expected_joins & predicted_joins), len(expected_joins)),
        },
    }


def extract_sql_structure(sql):
    parsed = parse_one(sql, dialect="sqlite")
    tables = {table.name for table in parsed.find_all(exp.Table)}
    columns = set()
    for column in parsed.find_all(exp.Column):
        table_name = column.table or ""
        column_name = column.name
        if column_name:
            columns.add((table_name, column_name))

    has_select_star = any(
        isinstance(select_exp, exp.Star)
        for select_exp in parsed.find_all(exp.Star)
    )
    return {
        "tables": sorted(tables),
        "columns": sorted(columns),
        "has_select_star": has_select_star,
    }


def evaluate_sql_case(case, prediction_row):
    sql = prediction_row["sql"] if isinstance(prediction_row, dict) else prediction_row
    structure = extract_sql_structure(sql)
    expected_tables = set(case.get("expected_tables", []))
    predicted_tables = set(structure["tables"])

    return {
        "candidate_sql": sql,
        "parsed_tables": structure["tables"],
        "table_coverage": _safe_ratio(len(expected_tables & predicted_tables), len(expected_tables)),
        "table_exact_match": expected_tables == predicted_tables,
        "has_select_star": structure["has_select_star"],
    }


def evaluate_execution_case(db_handler, case, prediction_row):
    sql = prediction_row["sql"] if isinstance(prediction_row, dict) else prediction_row
    reference_sql = case["reference_sql"]

    reference_result = db_handler.execute_query(reference_sql)
    normalized_reference = normalize_result_rows(reference_result)

    execution_report = {
        "candidate_sql": sql,
        "reference_row_count": len(normalized_reference),
    }

    if not SQLSecurityAudit.is_safe(sql):
        execution_report.update(
            {
                "executed": False,
                "safe": False,
                "result_exact_match": False,
                "result_subset_match": False,
                "error": "UNSAFE_SQL_BLOCKED",
            }
        )
        return execution_report

    try:
        candidate_result = db_handler.execute_query(sql)
        normalized_candidate = normalize_result_rows(candidate_result)
        reference_set = set(normalized_reference)
        candidate_set = set(normalized_candidate)

        execution_report.update(
            {
                "executed": True,
                "safe": True,
                "candidate_row_count": len(normalized_candidate),
                "result_exact_match": normalized_candidate == normalized_reference,
                "result_subset_match": candidate_set.issubset(reference_set),
                "result_overlap_ratio": _safe_ratio(len(reference_set & candidate_set), len(reference_set)),
            }
        )
        return execution_report
    except Exception as exc:
        execution_report.update(
            {
                "executed": False,
                "safe": True,
                "result_exact_match": False,
                "result_subset_match": False,
                "error": str(exc),
            }
        )
        return execution_report


def summarize_report(case_reports):
    if not case_reports:
        return {}

    retrieval_reports = [item["retrieval"] for item in case_reports]
    summary = {
        "case_count": len(case_reports),
        "avg_table_recall": round(sum(item["table_recall"] for item in retrieval_reports) / len(retrieval_reports), 4),
        "avg_column_recall": round(sum(item["column_recall"] for item in retrieval_reports) / len(retrieval_reports), 4),
        "avg_metric_recall": round(sum(item["metric_recall"] for item in retrieval_reports) / len(retrieval_reports), 4),
        "avg_join_recall": round(sum(item["join_recall"] for item in retrieval_reports) / len(retrieval_reports), 4),
        "table_exact_match_rate": round(sum(1 for item in retrieval_reports if item["table_exact_match"]) / len(retrieval_reports), 4),
    }

    sql_reports = [item["sql_structure"] for item in case_reports if "sql_structure" in item]
    if sql_reports:
        summary["sql_case_count"] = len(sql_reports)
        summary["avg_sql_table_coverage"] = round(
            sum(item["table_coverage"] for item in sql_reports) / len(sql_reports), 4
        )
        summary["sql_exact_match_rate"] = round(
            sum(1 for item in sql_reports if item["table_exact_match"]) / len(sql_reports), 4
        )
        summary["select_star_rate"] = round(
            sum(1 for item in sql_reports if item["has_select_star"]) / len(sql_reports), 4
        )

    execution_reports = [item["execution"] for item in case_reports if "execution" in item]
    if execution_reports:
        summary["execution_case_count"] = len(execution_reports)
        summary["execution_success_rate"] = round(
            sum(1 for item in execution_reports if item["executed"]) / len(execution_reports), 4
        )
        summary["result_exact_match_rate"] = round(
            sum(1 for item in execution_reports if item["result_exact_match"]) / len(execution_reports), 4
        )
        summary["result_subset_match_rate"] = round(
            sum(1 for item in execution_reports if item["result_subset_match"]) / len(execution_reports), 4
        )
    return summary
