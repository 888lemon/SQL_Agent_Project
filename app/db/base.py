from abc import ABC, abstractmethod

class BaseDBHandler(ABC):
    @abstractmethod
    def get_all_table_structures(self) -> dict:
        """获取所有表的 DDL"""
        pass

    @abstractmethod
    def get_schema_graph(self) -> dict:
        """获取表、字段和外键关系组成的结构化 Schema 图谱"""
        pass

    @abstractmethod
    def get_table_info(self, table_names: list = None) -> str:
        """获取指定表的结构信息"""
        pass

    @abstractmethod
    def execute_query(self, sql: str, max_rows: int = 200, timeout_seconds: float = 5.0) -> dict:
        """执行 SQL 并返回 {"columns": [], "data": []}"""
        pass
