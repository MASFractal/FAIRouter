import http.client
import io
import json
import urllib.error

import pytest

from fai_router.llm.client import FRACTALROUTER_URL, OPENROUTER_URL, LlmRequestError, OpenRouterClient


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _ok(text: str) -> FakeResponse:
    return FakeResponse(json.dumps({"choices": [{"message": {"content": text}}],
                                    "usage": {"prompt_tokens": 3, "completion_tokens": 5}}).encode())


def test_base_url_points_requests_at_the_provider(monkeypatch):
    seen = []

    def urlopen(request, timeout):
        seen.append((request.full_url, request.get_header("Authorization"), timeout))
        return _ok("ответ")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = OpenRouterClient("ключ", "m", base_url=FRACTALROUTER_URL + "/", timeout=7)
    assert client.complete([{"role": "user", "content": "привет"}]) == "ответ"
    assert seen == [(f"{FRACTALROUTER_URL}/chat/completions", "Bearer ключ", 7)]
    # Без base_url адрес прежний, OpenRouter
    assert OpenRouterClient("ключ", "m").url == f"{OPENROUTER_URL}/chat/completions"


def test_dropped_connection_is_retried(monkeypatch):
    """Обрыв соединения на стороне поставщика: запрос повторяется, ход не роняется."""
    attempts = []

    def urlopen(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return _ok("после повторов")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    client = OpenRouterClient("ключ", "m", retries=3)
    assert client.complete([{"role": "user", "content": "x"}]) == "после повторов"
    assert len(attempts) == 3


def test_error_after_retries_names_provider_and_model(monkeypatch):
    def urlopen(request, timeout):
        raise http.client.RemoteDisconnected("Remote end closed connection without response")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    client = OpenRouterClient("ключ", "openai/gpt-4o-mini", retries=2)
    with pytest.raises(LlmRequestError) as error:
        client.complete([{"role": "user", "content": "x"}])
    text = str(error.value)
    assert "openrouter.ai" in text and "openai/gpt-4o-mini" in text and "3 попыток" in text


def test_client_errors_are_not_retried(monkeypatch):
    """Неверный ключ или модель: повторять бессмысленно, ошибка уходит сразу с телом ответа."""
    attempts = []

    def urlopen(request, timeout):
        attempts.append(1)
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"error":"bad key"}'))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = OpenRouterClient("ключ", "m", retries=3)
    with pytest.raises(LlmRequestError, match="401.*bad key"):
        client.complete([{"role": "user", "content": "x"}])
    assert len(attempts) == 1


def test_overload_is_retried(monkeypatch):
    attempts = []

    def urlopen(request, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b""))
        return _ok("ok")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    assert OpenRouterClient("ключ", "m").complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(attempts) == 2
