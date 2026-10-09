"""Обращение к модели за ответом по схеме: свой бюджет времени, явный потолок ответа, один повтор
при обрезке или негодном ответе. Общее у распознавания задания, оценки стиля и судьи содержания,
как JsonCall в версии на C#.

Потолок ответа задан явно: разбор длинного ответа по схеме в прежние 3012 токенов не влезал. Обрезка
по потолку и ответ не по схеме это сбой с одним повтором, затем InvalidModelAnswer; исчерпанный
бюджет это TimeoutError; отказ поставщика (4xx) не повторяется."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, TypeVar

from fai_router.llm.client import OpenRouterClient

T = TypeVar("T")

# Потолок ответа модели в токенах
MAX_TOKENS = 8000

ATTEMPTS = 2


class InvalidModelAnswer(ValueError):
    """Модель ответила негодно: обрезанный JSON, ответ не по схеме или неполный вердикт."""


def ask(llm: Any, messages: list[dict[str, Any]], schema_name: str, schema: dict[str, Any],
        parse: Callable[[str], T | None], budget: float) -> T:
    """Ответ модели, разобранный по схеме. Обрезка по потолку и ответ, который разбор отверг,
    повторяются один раз, затем это InvalidModelAnswer. budget это бюджет времени в секундах на
    все попытки вместе: у клиента своего потолка на весь ход нет."""
    deadline = time.monotonic() + budget
    failure = ""
    for _ in range(ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Модель не ответила за {budget:.0f} с.")
        text, cut = _send(llm, messages, schema_name, schema, remaining)
        if cut:
            failure = "ответ обрезан по потолку токенов"
            continue
        result = parse_answer(text, parse)
        if result is not None:
            return result
        failure = "ответ не по схеме"
    raise InvalidModelAnswer(f"Модель ответила негодно на обеих попытках: {failure}.")


def extract_json(text: str | None) -> str | None:
    """JSON из ответа модели: без ограды ```json и текста вокруг. None, если фигурных скобок нет."""
    if not text or not text.strip():
        return None
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else None


def parse_answer(text: str | None, parse: Callable[[str], T | None]) -> T | None:
    """Разбор ответа: вырезанный JSON отдается разбору, его ошибка формата означает негодный ответ."""
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        return parse(raw)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _send(llm: Any, messages: list[dict[str, Any]], schema_name: str, schema: dict[str, Any],
          remaining: float) -> tuple[str, bool]:
    """Текст ответа и признак обрезки. Свой клиент получает остаток бюджета; клиент без полного
    ответа (только complete) обрезки не сообщает."""
    options = {"schema": schema, "schema_name": schema_name, "temperature": 0.0, "max_tokens": MAX_TOKENS}
    if isinstance(llm, OpenRouterClient):
        data = llm.complete_full(messages, budget=remaining, **options)
    elif hasattr(llm, "complete_full"):
        data = llm.complete_full(messages, **options)
    else:
        return str(llm.complete(messages, **options) or ""), False
    choice = ((data or {}).get("choices") or [{}])[0] or {}
    content = (choice.get("message") or {}).get("content")
    return (content if isinstance(content, str) else json.dumps(content) if content else ""), _is_cut(choice)


def _is_cut(choice: dict[str, Any]) -> bool:
    """Обрезка по потолку: у OpenAI-совместимых length, у Gemini MAX_TOKENS."""
    reasons = {str(choice.get("finish_reason") or "").lower(), str(choice.get("native_finish_reason") or "").lower()}
    return "length" in reasons or "max_tokens" in reasons
