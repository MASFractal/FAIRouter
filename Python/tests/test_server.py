"""Сервер, совместимый с OpenAI: токен, предел тела, массив частей в content, параметры клиента и
путь отзыва."""

import json
import logging
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from fai_router import FaiRouter, Settings
from fai_router.routed_element import RoutedElement
from fai_router.server import make_server
from tests.test_router import FakeLlm


def start(router, **kwargs):
    server = make_server(router, "127.0.0.1", 0, **kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def post(url, body, token=None, raw=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=raw if raw is not None else json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


@pytest.fixture
def recorded(tmp_path):
    """Роутер с исполнителем, который принимает параметры хода и запоминает их."""
    Settings.llm = FakeLlm()
    seen = []

    def execute(candidate, messages, options):
        seen.append((messages, options))
        return f"Ответ от {candidate.name}. Второе предложение."

    candidates = [RoutedElement("a", tps=100, dpmt_inp=1, dpmt_outp=2, ideal_match_vector=np.ones(Settings.full_dim()),
                                context_limit=30_000)]
    return FaiRouter(candidates, execute, database_path=str(tmp_path / "s.db"), measure=False), seen


def test_token_is_required_when_set(recorded):
    router, _ = recorded
    server, base = start(router, token="secret-1")
    try:
        status, data = post(f"{base}/chat/completions", {"messages": [{"role": "user", "content": "привет"}]})
        assert status == 401 and data["error"]["type"] == "authentication_error"
        status, _ = post(f"{base}/chat/completions", {"messages": [{"role": "user", "content": "привет"}]},
                         token="other")
        assert status == 401
        status, data = post(f"{base}/chat/completions", {"messages": [{"role": "user", "content": "привет"}]},
                            token="secret-1")
        assert status == 200 and data["model"] == "a"
    finally:
        server.shutdown()
        server.server_close()


def test_external_host_without_token_warns(recorded, caplog, monkeypatch):
    router, _ = recorded
    monkeypatch.delenv("FAI_ROUTER_TOKEN", raising=False)
    with caplog.at_level(logging.WARNING, logger="fai_router"):
        server = make_server(router, "0.0.0.0", 0)
        server.server_close()
    assert any("без токена" in record.getMessage() for record in caplog.records)


def test_body_size_is_limited(recorded):
    router, _ = recorded
    server, base = start(router, max_body=1000)
    try:
        status, data = post(f"{base}/chat/completions", None, raw=b"{" + b" " * 2000 + b"}")
        assert status == 413
        status, data = post(f"{base}/chat/completions", None, raw=b"[1, 2]")
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()


def test_content_parts_and_client_parameters_reach_the_executor(recorded):
    """content массивом частей склеивается по текстовым частям, а не роняет ход с 500; max_tokens и
    temperature клиента уходят исполнителю, исполнителю уходит весь диалог как есть."""
    router, seen = recorded
    server, base = start(router)
    messages = [{"role": "user", "content": [{"type": "text", "text": "Опиши картинку"},
                                             {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]}]
    try:
        status, data = post(f"{base}/chat/completions", {"messages": messages, "max_tokens": 321, "temperature": 0.7})
        assert status == 200 and data["choices"][0]["message"]["content"].startswith("Ответ от a")
        sent, options = seen[-1]
        assert sent == messages and options == {"max_tokens": 321, "temperature": 0.7}

        status, data = post(f"{base}/chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]}]})
        assert status == 400 and "сообщения пользователя" in data["error"]["message"]
    finally:
        server.shutdown()
        server.server_close()


def test_ordered_volume_sets_the_answer_ceiling(recorded):
    """Без max_tokens клиента потолок ответа следует из заказанного объема и не выше предела кандидата."""
    router, seen = recorded
    router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    assert seen[-1][1] == {"max_tokens": 4096, "temperature": 0.0}
    router.candidates[0].context_limit = 3000
    router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    assert seen[-1][1]["max_tokens"] == 1000


def test_feedback_path_and_saving_only_trained(recorded, tmp_path):
    router, _ = recorded
    server, base = start(router)
    try:
        status, data = post(f"{base}/chat/completions", {"messages": [{"role": "user", "content": "привет"}]})
        round_id = data["fai_router"]["round_id"]
        assert post(f"{base}/feedback", {"round_id": round_id, "score": 0.9}) == (200, {"ok": True, "round_id": round_id})
        assert post(f"{base}/feedback", {"round_id": round_id, "score": 7})[0] == 400
        assert post(f"{base}/feedback", {"round_id": 999, "score": 0.5})[0] == 400
        assert post(f"{base}/feedback", {"score": 0.5})[0] == 400
    finally:
        server.shutdown()
        server.server_close()
    # Ничего не обучено в этом процессе: при остановке сохранять нечего, чужие веса не затираются
    assert not router.unsaved
    assert router.traces.read_rated(router.candidates)[0].feedback.score == pytest.approx(0.9)


def test_failed_candidate_sits_out_the_cooldown(tmp_path):
    """Отказавший кандидат выпадает из выбора на failure_cooldown секунд."""
    Settings.llm = FakeLlm()
    calls = []

    def execute(candidate, messages):
        calls.append(candidate.name)
        if candidate.name == "сломанный":
            raise ConnectionError("поставщик недоступен")
        return "ответ"

    shared = np.ones(Settings.full_dim())
    candidates = [RoutedElement("сломанный", tps=500, dpmt_inp=0.1, dpmt_outp=0.1, ideal_match_vector=shared * 2),
                  RoutedElement("рабочий", tps=50, dpmt_inp=5, dpmt_outp=5, ideal_match_vector=shared)]
    router = FaiRouter(candidates, execute, measure=False, seed=1)
    from fai_router.settings import RouteWeights

    router.route_weights = RouteWeights(0.5, 0.25, 0.25, 0.0)
    first = router.ask("задача")
    assert first.winner == "рабочий" and first.trace.failed == ["сломанный"]
    second = router.ask("задача")
    assert second.trace.failed == [] and calls == ["сломанный", "рабочий", "рабочий"]

    router.failure_cooldown = 0
    router._failed_until.clear()
    assert router.ask("задача").trace.failed == ["сломанный"]


def test_seed_makes_the_draw_reproducible():
    Settings.llm = FakeLlm()

    def winners(seed):
        candidates = [RoutedElement(name, tps=100, dpmt_inp=1, dpmt_outp=1, ideal_match_vector=np.ones(Settings.full_dim()))
                      for name in ("a", "b", "c")]
        router = FaiRouter(candidates, lambda c, m: "ответ", measure=False, seed=seed)
        return [router.ask("Напиши обзор на 1500 знаков.").winner for _ in range(20)]

    assert winners(11) == winners(11)
    assert len(set(winners(11))) > 1
