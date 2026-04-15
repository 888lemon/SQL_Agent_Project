import unittest

from app.core.security import SQLSecurityAudit


class SQLSecurityAuditTestCase(unittest.TestCase):
    def test_allows_simple_select(self):
        self.assertTrue(SQLSecurityAudit.is_safe("SELECT * FROM Customers"))

    def test_blocks_multiple_statements(self):
        self.assertFalse(SQLSecurityAudit.is_safe("SELECT * FROM Customers; DROP TABLE Customers;"))

    def test_blocks_write_operations(self):
        self.assertFalse(SQLSecurityAudit.is_safe("DELETE FROM Customers"))

    def test_enforce_read_limit_for_select(self):
        bounded = SQLSecurityAudit.enforce_read_limit("SELECT * FROM Customers", max_rows=50)
        self.assertIn("LIMIT 50", bounded.upper())

    def test_enforce_read_limit_keeps_existing_limit(self):
        sql = "SELECT * FROM Customers LIMIT 10"
        bounded = SQLSecurityAudit.enforce_read_limit(sql, max_rows=50)
        self.assertEqual(sql, bounded)


if __name__ == "__main__":
    unittest.main()
