# app/core/security.py
from sqlglot import exp, parse

class SQLSecurityAudit:
    """基于 AST 的静态 SQL 审计引擎"""

    # 定义黑名单操作
    FORBIDDEN_OPERATIONS = (exp.Delete, exp.Drop, exp.Update, exp.Alter, exp.Insert)

    @staticmethod
    def is_safe(sql_query: str) -> bool:
        try:
            statements = parse(sql_query, dialect="sqlite")
            if len(statements) != 1:
                print("⚠️ 安全拦截：仅允许执行单条 SQL 语句")
                return False

            parsed = statements[0]

            for node in parsed.walk():
                if isinstance(node, SQLSecurityAudit.FORBIDDEN_OPERATIONS):
                    print(f"⚠️ 安全拦截：发现非法指令 {type(node)}")
                    return False

            for table in parsed.find_all(exp.Table):
                if "SQLITE_" in table.name.upper():
                    return False

            return True
        except Exception as e:
            print(f"❌ 语法分析失败: {e}")
            return False

    @staticmethod
    def enforce_read_limit(sql_query: str, max_rows: int) -> str:
        """
        为只读查询补充 LIMIT，避免一次性返回超大结果集。
        非 SELECT / WITH 查询保持原样，由 is_safe 决定是否拦截。
        """
        sql = sql_query.strip().rstrip(";")
        if not sql:
            raise ValueError("SQL 不能为空。")

        lowered = sql.lower()
        if lowered.startswith("select") or lowered.startswith("with"):
            if " limit " in f" {lowered} ":
                return sql
            return f"{sql} LIMIT {max_rows}"
        return sql

# 测试一下
if __name__ == "__main__":
    test_sql = "SELECT * FROM stu_info; DROP TABLE stu_info;"
    print(f"SQL 安全吗？ {SQLSecurityAudit.is_safe(test_sql)}")
