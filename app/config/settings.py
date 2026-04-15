import os
from dataclasses import dataclass


PLACEHOLDER_API_KEYS = {"", "your_api_key_here", None}


@dataclass
class AppSettings:
    deepseek_api_key: str | None
    deepseek_base_url: str
    db_type: str
    db_path: str
    embedding_model_name: str
    cache_threshold: float
    cache_ttl_seconds: int
    cache_max_entries: int

    @property
    def has_valid_api_key(self) -> bool:
        return self.deepseek_api_key not in PLACEHOLDER_API_KEYS


def load_settings() -> AppSettings:
    return AppSettings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        db_type=os.getenv("DB_TYPE", "sqlite"),
        db_path=os.getenv("DB_PATH", "app/data/northwind.db"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        cache_threshold=float(os.getenv("CACHE_THRESHOLD", "0.95")),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "1800")),
        cache_max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "500")),
    )


def validate_required_settings(settings: AppSettings):
    if not settings.has_valid_api_key:
        raise ValueError("未检测到可用的 DEEPSEEK_API_KEY，请在 .env 中填写真实 API Key。")
