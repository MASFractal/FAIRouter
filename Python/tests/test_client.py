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


def test_provider_is_recognized_by_key_shape(monkeypatch, tmp_path):
    from fai_router.llm.client import base_url_for_key, provider_from_environment

    assert base_url_for_key("sk-or-v1-abc") == OPENROUTER_URL
    assert base_url_for_key("rtr_live_abc") == FRACTALROUTER_URL
    assert base_url_for_key("frr_test_abc") == FRACTALROUTER_URL
    # Незнакомый ключ не уходит чужому поставщику
    for foreign in ("sk-proj-abc", "sk-ant-abc", "abc"):
        with pytest.raises(ValueError, match="не опознан"):
            base_url_for_key(foreign)

    monkeypatch.delenv("FRACTALROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert provider_from_environment(tmp_path) == (FRACTALROUTER_URL, "")

    (tmp_path / "key.txt").write_text("sk-or-v1-file\n", encoding="utf-8")
    assert provider_from_environment(tmp_path) == (OPENROUTER_URL, "sk-or-v1-file")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert provider_from_environment(tmp_path) == (OPENROUTER_URL, "sk-or-env")
    # Ключ FractalRouter в окружении главнее: наш поставщик первый
    monkeypatch.setenv("FRACTALROUTER_API_KEY", " rtr_live_env ")
    assert provider_from_environment(tmp_path) == (FRACTALROUTER_URL, "rtr_live_env")


def test_budget_bounds_retries(monkeypatch):
    """Срок на все попытки: исчерпан, тогда TimeoutError, а не еще одна попытка."""
    attempts = []
    clock = [0.0]

    def urlopen(request, timeout):
        attempts.append(timeout)
        clock[0] += 5.0
        raise http.client.RemoteDisconnected("обрыв")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    client = OpenRouterClient("ключ", "m", retries=5, timeout=100)
    with pytest.raises(TimeoutError):
        client.complete_full([{"role": "user", "content": "x"}], budget=8)
    assert attempts == [8, 3]


def test_none_parameters_are_not_sent(monkeypatch):
    bodies = []

    def urlopen(request, timeout):
        bodies.append(json.loads(request.data))
        return _ok("ответ")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    OpenRouterClient("ключ", "m").complete_full([{"role": "user", "content": "x"}], temperature=None, max_tokens=None)
    assert "temperature" not in bodies[0] and "max_tokens" not in bodies[0]
