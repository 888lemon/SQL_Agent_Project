import argparse
import json
from pathlib import Path

from app.eval.evaluator import (
    build_db_handler,
    build_retriever,
    evaluate_execution_case,
    evaluate_retrieval_case,
    evaluate_sql_case,
    load_dataset,
    load_predictions,
    summarize_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline RAG and SQL structure evaluation.")
    parser.add_argument("--dataset", default="app/data/eval_dataset.json", help="Path to evaluation dataset JSON.")
    parser.add_argument("--db-path", default="app/data/northwind.db", help="Path to SQLite database.")
    parser.add_argument("--metadata-path", default="app/schema/metadata.yaml", help="Path to metadata YAML.")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Embedding model name.")
    parser.add_argument("--predictions", help="Optional JSON or JSONL file with generated SQL predictions.")
    parser.add_argument("--output", help="Optional path to write the evaluation report JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset)
    predictions = load_predictions(args.predictions)
    retriever = build_retriever(args.db_path, args.metadata_path, args.model_name)
    db_handler = build_db_handler(args.db_path)

    case_reports = []
    for case in dataset:
        report = evaluate_retrieval_case(retriever, case)
        prediction_row = predictions.get(case["id"])
        if prediction_row and prediction_row.get("sql"):
            report["sql_structure"] = evaluate_sql_case(case, prediction_row)
            report["execution"] = evaluate_execution_case(db_handler, case, prediction_row)
        else:
            report["execution"] = evaluate_execution_case(
                db_handler,
                case,
                {"sql": case["reference_sql"]},
            )
        case_reports.append(report)

    final_report = {
        "dataset": args.dataset,
        "summary": summarize_report(case_reports),
        "cases": case_reports,
    }

    print(json.dumps(final_report["summary"], ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(final_report, file, ensure_ascii=False, indent=2)
        print(f"Detailed report written to: {output_path}")


if __name__ == "__main__":
    main()
