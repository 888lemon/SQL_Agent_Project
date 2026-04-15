import logging
import json
import os
from datetime import datetime


def setup_metrics_logger(log_file="metrics.log"):
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("MetricsLogger")
    logger.setLevel(logging.INFO)

    # 避免重复添加 Handler
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        # 定义格式：仅记录消息内容（因为我们会手动构建 JSON）
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


metrics_logger = setup_metrics_logger()


def log_metrics(question, metrics_dict):
    """将度量数据以 JSON 行的形式写入日志"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        **metrics_dict
    }
    # 写入一行 JSON
    metrics_logger.info(json.dumps(log_entry, ensure_ascii=False))