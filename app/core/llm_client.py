import time
from dataclasses import dataclass

import httpx


@dataclass
class LLMCallResult:
    content: str | None
    error_type: str | None = None
    error_message: str | None = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.content is not None


class LLMService:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

    def _request_once(self, payload: dict) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, verify=True) as client:
            response = client.post("/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.ConnectError):
            return "network_error"
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403):
                return "auth_error"
            if status == 429:
                return "rate_limit"
            if 500 <= status < 600:
                return "server_error"
            return "http_error"
        return "unknown_error"

    @staticmethod
    def _is_retryable(error_type: str) -> bool:
        return error_type in {"timeout", "network_error", "rate_limit", "server_error"}

    def call(
        self,
        prompt: str,
        *,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_error: Exception | None = None
        last_error_type = "unknown_error"

        for attempt in range(1, self.max_retries + 1):
            try:
                content = self._request_once(payload)
                return LLMCallResult(content=content, attempts=attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                last_error_type = self._classify_error(exc)
                if attempt >= self.max_retries or not self._is_retryable(last_error_type):
                    break
                sleep_seconds = self.backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)

        return LLMCallResult(
            content=None,
            error_type=last_error_type,
            error_message=str(last_error) if last_error else "unknown failure",
            attempts=self.max_retries,
        )
