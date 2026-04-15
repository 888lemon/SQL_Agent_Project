from .handler import SQLiteHandler, MySQLHandler

class DBFactory:
    @staticmethod
    def get_handler(db_type: str, config: any):
        """
        根据类型生产对应的处理器
        :param db_type: 'sqlite' 或 'mysql'
        :param config: sqlite 为路径字符串，mysql 为配置字典
        """
        db_type = db_type.lower()
        if db_type == "sqlite":
            return SQLiteHandler(db_path=config)
        elif db_type == "mysql":
            return MySQLHandler(config=config)
        else:
            raise ValueError(f"❌ 不支持的数据库驱动类型: {db_type}")