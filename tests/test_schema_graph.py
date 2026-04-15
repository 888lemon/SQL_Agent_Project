import unittest

from app.db.handler import SQLiteHandler


class SQLiteSchemaGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.handler = SQLiteHandler("app/data/northwind.db")

    def test_schema_graph_contains_tables_and_joins(self):
        graph = self.handler.get_schema_graph()
        self.assertIn("Orders", graph["tables"])
        self.assertIn("Order Details", graph["tables"])
        self.assertTrue(graph["joins"])

    def test_orders_table_has_expected_foreign_keys(self):
        graph = self.handler.get_schema_graph()
        orders = graph["tables"]["Orders"]
        targets = {fk["to_table"] for fk in orders["foreign_keys"]}
        self.assertIn("Customers", targets)
        self.assertIn("Employees", targets)
        self.assertIn("Shippers", targets)


if __name__ == "__main__":
    unittest.main()
