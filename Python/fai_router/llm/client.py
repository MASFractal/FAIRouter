from __future__ import annotations

import json
import urllib.request
from typing import Any


class OpenRouterClient:
    """Клиент к моделям через OpenRouter по протоколу chat completions со структурированным
    ответом по схеме. Только стандартная библиотека: в C# это делал клиент из AI.LLM, здесь
    хватает urllib."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: float = 120.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete_full(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        schema_name: str = "answer",
        temperature: float = 0.0,
        max_tokens: int = 3012,
    ) -> dict[str, Any]:
        """Полный ответ поставщика, включая расход токенов."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        request = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Текст ответа модели."""
        data = self.complete_full(messages, **kwargs)
        return data["choices"][0]["message"]["content"] or ""

    @staticmethod
    def usage(data: dict[str, Any]) -> tuple[int, int]:
        """Токены входа и выхода из полного ответа."""
        usage = data.get("usage") or {}
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
