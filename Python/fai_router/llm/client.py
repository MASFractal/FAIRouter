from __future__ import annotations

import http.client
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("fai_router")

# Адреса поставщиков по протоколу OpenAI chat completions. FractalRouter (fractalrouter.ru)
# это наш шлюз к моделям с оплатой в рублях; любой другой поставщик с тем же протоколом
# подключается через base_url
OPENROUTER_URL = "https://openrouter.ai/api/v1"
FRACTALROUTER_URL = "https://api.fractalrouter.ru/v1"

# Ответы поставщика, после которых запрос стоит повторить: перегрузка и сбои на его стороне
_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LlmRequestError(RuntimeError):
    """Запрос к модели не удался и после повторов. В сообщении поставщик, модель и причина:
    по голому «Remote end closed connection» не понять, кто и на чем оборвал связь."""


class OpenRouterClient:
    """Клиент к моделям по протоколу OpenAI chat completions со структурированным ответом по
    схеме. Имя историческое: первым поставщиком был OpenRouter, но протокол тот же у FractalRouter
    и у любого совместимого сервера, адрес задается base_url. Только стандартная библиотека:
    в C# это делал клиент из AI.LLM, здесь хватает urllib.

    Обрыв соединения, таймаут и перегрузка поставщика повторяются retries раз с растущей
    паузой: одиночный сбой сети не должен ронять ход, за который исполнитель уже ответил."""

    URL = f"{OPENROUTER_URL}/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: float = 120.0,
                 base_url: str = OPENROUTER_URL, retries: int = 3, retry_pause: float = 1.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.retries = retries
        self.retry_pause = retry_pause

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
        data = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                pause = self.retry_pause * 2 ** (attempt - 1)
                log.warning("%s: повтор %d из %d через %.0f с после ошибки: %s",
                            self._where(), attempt, self.retries, pause, last)
                time.sleep(pause)
            try:
                return self._post(data)
            except urllib.error.HTTPError as error:
                if error.code not in _RETRY_STATUSES:
                    raise LlmRequestError(f"{self._where()}: поставщик ответил {error.code} {error.reason}: "
                                          f"{_read_error(error)}") from error
                last = error
            except (http.client.RemoteDisconnected, http.client.IncompleteRead, ConnectionError,
                    socket.timeout, TimeoutError, urllib.error.URLError) as error:
                last = error
        raise LlmRequestError(
            f"{self._where()}: связь не удалась после {self.retries + 1} попыток: {last}. "
            "Если ошибка повторяется, поставщик недоступен из вашей сети (блокировка или прокси); "
            "попробуйте другой base_url, например FaiRouter.from_fractalrouter.") from last

    def _post(self, data: bytes) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "X-Title": "FAIRouter"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _where(self) -> str:
        return f"{self.base_url} ({self.model})"

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Текст ответа модели."""
        data = self.complete_full(messages, **kwargs)
        return data["choices"][0]["message"]["content"] or ""

    @staticmethod
    def usage(data: dict[str, Any]) -> tuple[int, int]:
        """Токены входа и выхода из полного ответа."""
        usage = data.get("usage") or {}
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def _read_error(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", "replace")[:500]
    except Exception:  # noqa: BLE001 - тело ошибки необязательно
        return ""
