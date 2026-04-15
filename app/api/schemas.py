from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str | None = None
    text: str | None = None

    def resolved_question(self) -> str:
        """兼容历史字段 text，统一转换为 question。"""
        raw = self.question if self.question is not None else self.text
        if raw is None or not raw.strip():
            raise ValueError("请求体必须包含非空的 question 或 text 字段。")
        return raw.strip()
