import json
import threading
import urllib.request

import numpy as np
import pytest

from fai_router import FaiRouter, Settings
from fai_router.enums import Capability, Style
from fai_router.routed_element import RoutedElement
from fai_router.server import make_server


class FakeLlm:
    """Подмена модели без сети: отдает готовые JSON по имени схемы. Детские задачи отличает
    по слову в запросе, иначе все задачи журнала совпали бы, вычитание среднего дало бы ноль
    и обучению было бы не от чего двигаться."""

    def complete(self, messages, schema=None, schema_name="answer", **kwargs):
        childish = "ребен" in messages[-1]["content"].lower()
        if schema_name == "input_specifications":
            if childish:
                return json.dumps({"styleType": "Children", "symbolLength": 600, "wordLength": 90,
                                   "paragraphCount": 3, "sectionCount": 0, "listItemCount": 0, "tableCount": 0,
                                   "codeBlockCount": 0, "formulaCount": 0, "headingDepth": 0,
                                   "avgSentenceLength": 8, "readabilityScore": 90, "termDensity": 0.05,
                                   "formalityScore": 0.1, "language": "ru", "hasReferences": False})
            return json.dumps({"styleType": "Scientific", "symbolLength": 1500, "wordLength": 220,
                               "paragraphCount": 4, "sectionCount": 3, "listItemCount": 0, "tableCount": 1,
                               "codeBlockCount": 0, "formulaCount": 0, "headingDepth": 2,
                               "avgSentenceLength": 18, "readabilityScore": 35, "termDensity": 0.7,
                               "formalityScore": 0.9, "language": "ru", "hasReferences": True})
        return json.dumps({"styleType": "Scientific", "termDensity": 0.65, "formalityScore": 0.85})


def make_router(tmp_path, measure=True):
    Settings.llm = FakeLlm()
    shared = np.ones(Settings.full_dim())
    candidates = [RoutedElement("cheap", tps=200, dpmt_inp=0.3, dpmt_outp=2.5, ideal_match_vector=shared.copy()),
                  RoutedElement("strong", tps=60, dpmt_inp=1.0, dpmt_outp=5.0, ideal_match_vector=shared.copy())]
    calls = []

    def execute(candidate, messages):
        calls.append((candidate.name, messages[-1]["content"]))
        return f"# Ответ от {candidate.name}\n\nПервый абзац про кластеризацию. Второй абзац с выводом."

    router = FaiRouter(candidates, execute, database_path=str(tmp_path / "r.db"), measure=measure)
    return router, calls


def test_ask_runs_whole_loop_and_journals(tmp_path):
    router, calls = make_router(tmp_path)
    answer = router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")

    assert answer.winner in ("cheap", "strong")
    assert calls[0][0] == answer.winner
    assert answer.text.startswith("# Ответ от")
    assert answer.requested.style_type == Style.SCIENTIFIC
    assert answer.actual is not None and answer.critic is not None
    assert 0 <= answer.score <= 1
    assert answer.round_id == 1
    assert router.traces.count() == (1, 1)


def test_feedback_train_save_load(tmp_path):
    router, _ = make_router(tmp_path)
    router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    router.ask("Объясни ребенку, что такое кластеризация, в 600 знаков.")
    router.ask("Напиши научный обзор методов регуляризации на 1500 знаков.")
    router.feedback(1, 1.0)
    before = [c.ideal_match_vector.copy() for c in router.candidates]

    loss = router.train(epochs=5)
    assert loss >= 0
    assert Settings.task_mean is not None
    assert any(not np.array_equal(b, c.ideal_match_vector) for b, c in zip(before, router.candidates))

    router.save()
    Settings.task_mean = None
    again = FaiRouter(router.candidates, lambda c, m: "", database_path=str(tmp_path / "r.db"), measure=False)
    assert Settings.task_mean is not None
    # Три хода оценены, но в опыт идет только человеческий отзыв, а он один: автоотзыв
    # температуру не сбивает, разведка не гаснет по мнению собственного судьи
    assert again.candidates[0].experience + again.candidates[1].experience == 1


def test_capability_requirement_reaches_router(tmp_path):
    router, calls = make_router(tmp_path, measure=False)
    router.candidates[0].capabilities = Capability.CODE
    answer = router.ask("Задача с картинкой", required=Capability.VISION)
    assert answer.winner == "strong"


def test_openai_compatible_server(tmp_path):
    router, _ = make_router(tmp_path, measure=False)
    server = make_server(router, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        with urllib.request.urlopen(f"{base}/models") as response:
            models = json.load(response)
        assert [m["id"] for m in models["data"]] == ["auto", "cheap", "strong"]

        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps({"model": "auto", "messages": [
                {"role": "system", "content": "Ты помощник."},
                {"role": "user", "content": "Напиши научный обзор на 1500 знаков."}]}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer anything"})
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
        assert data["object"] == "chat.completion"
        assert data["model"] in ("cheap", "strong")
        assert data["choices"][0]["message"]["content"].startswith("# Ответ от")
        assert data["fai_router"]["round_id"] == 1

        streamed = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps({"model": "auto", "stream": True,
                             "messages": [{"role": "user", "content": "Еще раз на 1500 знаков."}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(streamed) as response:
            text = response.read().decode()
        assert text.startswith("data: ") and text.rstrip().endswith("data: [DONE]")
    finally:
        server.shutdown()
        server.server_close()


def test_len_answer_follows_recognized_order(tmp_path):
    """Объем ответа берется из распознанного заказа, а не из длины промпта: короткая просьба о
    длинном тексте иначе выглядела дешевой и быстрой у всех кандидатов разом."""
    from fai_router.services import InputFeaturesService

    make_router(tmp_path, measure=False)
    short_prompt = "Обзор на 1500 знаков."
    features = InputFeaturesService.get_features_full(short_prompt)
    assert features.input_specifications.symbol_length == 1500
    assert features.len_answer == 1500 / InputFeaturesService.EST_SYMBOL_PER_TOKEN
    assert features.len_answer > InputFeaturesService.get_features(short_prompt).len_answer


def test_judge_learns_only_from_human_feedback(tmp_path):
    """После хода стоит автоотзыв; обучение по нему не должно трогать судью, иначе одна
    автоматическая оценка подгонялась бы под другую. Человеческий отзыв судью двигает."""
    router, _ = make_router(tmp_path)
    router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    before = router.judge.transformer_w.copy()
    router.train()
    assert np.array_equal(before, router.judge.transformer_w)

    router.feedback(1, 0.0)
    router.train()
    assert not np.array_equal(before, router.judge.transformer_w)


def test_from_openai_compatible_builds_candidates_without_catalog(monkeypatch):
    """Цены заданы руками: каталог OpenRouter не нужен, сеть не трогается, а снимок замеров
    дает начальные веса. Исполнитель и судья ходят на base_url поставщика."""
    from fai_router import catalog

    monkeypatch.setattr(catalog, "fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("сеть не нужна")))
    router = FaiRouter.from_openai_compatible(
        "https://example.test/v1", "ключ", ["anthropic/claude-opus-4.7", "my/own-model"],
        prices={"anthropic/claude-opus-4.7": (15.0, 75.0), "my/own-model": (1.0, 2.0)},
        tokens_per_second={"my/own-model": 120}, measure=False)

    names = [c.name for c in router.candidates]
    assert names == ["anthropic/claude-opus-4.7", "my/own-model"]
    assert router.candidates[1].dpmt_inp == 1.0 and router.candidates[1].dpmt_outp == 2.0
    assert router.candidates[1].tps == 120
    assert Settings.llm.base_url == "https://example.test/v1"
    assert Settings.llm.model == "openai/gpt-4o-mini"
    # Начальный вектор известной модели пришел из снимка замеров, а не из случайного Ксавье
    from fai_router.benchmarks import default_snapshot
    from fai_router.training import benchmark_prior
    expected = benchmark_prior.vector(default_snapshot(), "anthropic/claude-opus-4.7")
    assert expected is not None
    assert np.array_equal(router.candidates[0].ideal_match_vector, expected)


def test_from_fractalrouter_uses_its_own_catalog_and_address(monkeypatch):
    """FractalRouter: каталог свой, по ключу, в рублях; OpenRouter не нужен вовсе."""
    from fai_router import catalog
    from fai_router.llm.client import FRACTALROUTER_URL

    monkeypatch.setattr(catalog, "fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("OpenRouter не нужен")))
    seen = []

    def fetch_fractalrouter(api_key, timeout=60.0):
        seen.append(api_key)
        return catalog.parse_fractalrouter(json.dumps({"data": [
            {"id": "anthropic/claude-sonnet-5", "owned_by": "Anthropic", "context_length": 1000000,
             "modalities": ["text", "vision"],
             "pricing": {"prompt_rub_per_1m": "213.55", "completion_rub_per_1m": "1067,74"}},
            {"id": "openai/gpt-4o-mini", "owned_by": "OpenAI", "context_length": 128000,
             "modalities": ["text"], "pricing": {"prompt_rub_per_1m": "12", "completion_rub_per_1m": "48"}},
        ]}))

    monkeypatch.setattr(catalog, "fetch_fractalrouter", fetch_fractalrouter)
    router = FaiRouter.from_fractalrouter("rtr_live_x", ["anthropic/claude-sonnet-5", "openai/gpt-4o-mini"],
                                          measure=False)
    assert seen == ["rtr_live_x"]
    assert Settings.llm.base_url == FRACTALROUTER_URL and Settings.llm.api_key == "rtr_live_x"
    sonnet, mini = router.candidates
    assert (sonnet.dpmt_inp, sonnet.dpmt_outp) == (213.55, 1067.74)
    assert sonnet.capabilities & Capability.VISION and not (mini.capabilities & Capability.VISION)
    assert sonnet.context_limit == 0


def test_unknown_model_without_price_is_an_error(monkeypatch):
    from fai_router import catalog

    monkeypatch.setattr(catalog, "fetch", lambda *a, **k: [])
    with pytest.raises(ValueError, match="нет цены"):
        FaiRouter.from_openai_compatible("https://example.test/v1", "ключ", ["nobody/knows"], measure=False)


def test_measurement_failure_keeps_the_answer(tmp_path):
    """Судья отвалился по сети: ответ исполнителя уже получен и возвращается без оценки, ход
    записан в журнал без автоотзыва."""
    router, _ = make_router(tmp_path)

    class BrokenLlm(FakeLlm):
        def complete(self, messages, schema=None, schema_name="answer", **kwargs):
            if schema_name != "input_specifications":
                raise ConnectionError("Remote end closed connection without response")
            return super().complete(messages, schema, schema_name, **kwargs)

    Settings.llm = BrokenLlm()
    answer = router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    assert answer.text.startswith("# Ответ от")
    assert answer.score is None and answer.actual is None and answer.assessment is None
    assert answer.round_id == 1
    assert router.traces.count() == (1, 0)


def test_feedback_must_be_between_zero_and_one(tmp_path):
    router, _ = make_router(tmp_path)
    router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    router.feedback(1, 0.2)
    with pytest.raises(ValueError, match="от 0"):
        router.feedback(1, 5)


def test_named_model_sets_and_profiles(monkeypatch):
    """«popular» берет популярные из комплекта, которые есть в каталоге, «all» весь каталог; список
    требует цены у каждого. Профиль весов именуется словом и действует на ход."""
    from fai_router import catalog
    from fai_router.settings import RouteWeights

    popular = catalog.popular_models()
    assert 40 <= len(popular) <= 80 and "openai/gpt-4o-mini" in popular

    fake = [catalog.ModelInfo(popular[0], "a", 1, 2, 0, 0, Capability.CODE),
            catalog.ModelInfo("vendor/unknown", "b", 3, 4, 0, 0, Capability.CODE),
            catalog.ModelInfo("vendor/broken", "c", -1, -1, 0, 0, Capability.CODE)]
    monkeypatch.setattr(catalog, "fetch", lambda *a, **k: fake)

    by_popular = FaiRouter.from_openai_compatible("https://example.test/v1", "k", "popular", measure=False)
    assert [c.name for c in by_popular.candidates] == [popular[0]]

    by_all = FaiRouter.from_openai_compatible("https://example.test/v1", "k", "all", measure=False, weights="price")
    assert sorted(c.name for c in by_all.candidates) == sorted([popular[0], "vendor/unknown"])
    assert by_all.route_weights == RouteWeights.price()

    with pytest.raises(ValueError, match="нет цены"):
        FaiRouter.from_openai_compatible("https://example.test/v1", "k", ["vendor/missing"], measure=False)
    with pytest.raises(ValueError, match="профиль"):
        FaiRouter.from_openai_compatible("https://example.test/v1", "k", "all", measure=False, weights="fastest")

    assert RouteWeights.profile("quality").WQ == 0.8 and RouteWeights.profile("balance").WC == 0.25
    assert RouteWeights.profile(None) is None


def test_profile_reaches_the_route(tmp_path, monkeypatch):
    from fai_router import env
    from fai_router.settings import RouteWeights

    router, _ = make_router(tmp_path, measure=False)
    seen = []
    original = env.route

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights"))
        return original(*args, **kwargs)

    monkeypatch.setattr(env, "route", spy)
    router.ask("Напиши научный обзор на 1500 знаков.")
    router.ask("Напиши научный обзор на 1500 знаков.", weights="quality")
    assert seen == [None, RouteWeights.quality()]
