import sqlite3
import time
from .base import BaseDBHandler
from app.core.security import SQLSecurityAudit

class SQLiteHandler(BaseDBHandler):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def get_all_table_structures(self) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows if row[1]}

    def get_schema_graph(self) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()

        schema = {"tables": {}, "joins": []}
        for table_name, ddl in tables:
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = []
            primary_keys = []
            for row in cursor.fetchall():
                column_info = {
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5])
                }
                columns.append(column_info)
                if row[5]:
                    primary_keys.append(row[1])

            cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
            foreign_keys = []
            for row in cursor.fetchall():
                fk = {
                    "from": row[3],
                    "to_table": row[2],
                    "to": row[4]
                }
                foreign_keys.append(fk)
                schema["joins"].append({
                    "left_table": table_name,
                    "left_column": row[3],
                    "right_table": row[2],
                    "right_column": row[4]
                })

            schema["tables"][table_name] = {
                "ddl": ddl,
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys
            }

        conn.close()
        return schema

    def get_table_info(self, table_names: list = None) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        if table_names:
            placeholders = ', '.join(['?'] * len(table_names))
            query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({placeholders});"
            cursor.execute(query, table_names)
        else:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        schemas = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return "\n\n".join(schemas)

    def execute_query(self, sql: str, max_rows: int = 200, timeout_seconds: float = 5.0) -> dict:
        # 使用 pandas 简化处理，自动获取列名；延迟导入以避免非查询路径受环境问题影响
        import pandas as pd

        conn = self._get_conn()
        bounded_sql = SQLSecurityAudit.enforce_read_limit(sql, max_rows=max_rows)
        deadline = time.monotonic() + timeout_seconds

        def _progress_handler():
            # 返回非 0 表示中断当前 SQLite 查询
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(_progress_handler, 10_000)
        try:
            df = pd.read_sql_query(bounded_sql, conn)
            return {
                "columns": list(df.columns),
                "data": df.values.tolist()
            }
        except Exception as e:
            if "interrupted" in str(e).lower():
                raise TimeoutError(f"SQL 查询超时（>{timeout_seconds}s）") from e
            raise e
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()

class MySQLHandler(BaseDBHandler):
    """预留 MySQL 实现，展示架构扩展性"""
    def __init__(self, config: dict):
        self.config = config # 包含 host, user, password 等

    def get_all_table_structures(self) -> dict:
        # 逻辑示例：查询 INFORMATION_SCHEMA.TABLES
        return {"example_table": "CREATE TABLE ..."}

    def get_schema_graph(self) -> dict:
        return {"tables": {}, "joins": []}

    def get_table_info(self, table_names: list = None) -> str:
        return "MySQL Specific DDL"

    def execute_query(self, sql: str, max_rows: int = 200, timeout_seconds: float = 5.0) -> dict:
        return {"columns": [], "data": []}
