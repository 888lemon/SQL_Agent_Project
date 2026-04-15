import unittest

from app.eval.evaluator import (
    evaluate_execution_case,
    extract_sql_structure,
    normalize_result_rows,
    summarize_report,
)
from app.db.handler import SQLiteHandler


class EvaluatorTestCase(unittest.TestCase):
    def setUp(self):
        self.db_handler = SQLiteHandler("app/data/northwind.db")

    def test_extract_sql_structure(self):
        structure = extract_sql_structure(
            "SELECT c.CompanyName, COUNT(o.OrderID) FROM Customers c "
            "JOIN Orders o ON c.CustomerID = o.CustomerID GROUP BY c.CompanyName"
        )
        self.assertIn("Customers", structure["tables"])
        self.assertIn("Orders", structure["tables"])
        self.assertFalse(structure["has_select_star"])

    def test_normalize_result_rows(self):
        result = {
            "columns": ["a", "b"],
            "data": [[1.23456789, None], [1, "x"]],
        }
        normalized = normalize_result_rows(result)
        self.assertEqual(normalized[0], ("1", "x"))
        self.assertEqual(normalized[1], ("1.234568", None))

    def test_evaluate_execution_case_exact_match(self):
        case = {
            "reference_sql": "SELECT COUNT(*) AS cnt FROM Orders",
        }
        prediction = {"sql": "SELECT COUNT(*) AS cnt FROM Orders"}
        report = evaluate_execution_case(self.db_handler, case, prediction)
        self.assertTrue(report["executed"])
        self.assertTrue(report["result_exact_match"])

    def test_summarize_report(self):
        summary = summarize_report(
            [
                {
                    "retrieval": {
                        "table_recall": 1.0,
                        "column_recall": 0.5,
                        "metric_recall": 1.0,
                        "join_recall": 1.0,
                        "table_exact_match": True,
                    }
                },
                {
                    "retrieval": {
                        "table_recall": 0.5,
                        "column_recall": 1.0,
                        "metric_recall": 0.0,
                        "join_recall": 1.0,
                        "table_exact_match": False,
                    },
                    "sql_structure": {
                        "table_coverage": 1.0,
                        "table_exact_match": True,
                        "has_select_star": False,
                    },
                    "execution": {
                        "executed": True,
                        "result_exact_match": True,
                        "result_subset_match": True,
                    },
                },
            ]
        )
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["avg_table_recall"], 0.75)
        self.assertEqual(summary["sql_case_count"], 1)
        self.assertEqual(summary["execution_case_count"], 1)


if __name__ == "__main__":
    unittest.main()
