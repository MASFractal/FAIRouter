"""Снимок внешних замеров: сопоставление имен, качество в серии, начальный вектор, скорость и цена.

Ожидаемые числа общие с C#-тестом (Mas.Core.Tests/RouterBenchmarkTests.cs): расхождение версий
видно по расхождению чисел."""

import json

import numpy as np
import pytest

from fai_router import catalog
from fai_router.benchmarks import BenchmarkEntry, BenchmarkSnapshot, canonical, find, slug
from fai_router.settings import Settings
from fai_router.training import benchmark_prior

# Серия парных предпочтений: оценка это рейтинг Эло, у записи есть число голосов
PREFERENCE_ROWS = [
    BenchmarkEntry("claude-opus-4-7-thinking", "claude-opus-4-7-high", "Anthropic", 1540.9, 24222),
    BenchmarkEntry("claude-opus-4-7-text", "claude-opus-4-7", "Anthropic", 1520.5, 30000),
    BenchmarkEntry("gemini-3-flash-text", "gemini-3-flash", "Google", 1400.0, 9000),
    BenchmarkEntry("gpt-5-2-2025-12-11-text", "gpt-5.2-2025-12-11", "OpenAI", 1300.0, 100),
]


def snapshot() -> BenchmarkSnapshot:
    literary = [BenchmarkEntry("claude-opus-4-7-text", "claude-opus-4-7", "Anthropic", 1400.0, 100),
                BenchmarkEntry("gemini-3-flash-text", "gemini-3-flash", "Google", 1500.0, 100)]
    return BenchmarkSnapshot("2026-09-14T00:00:00+00:00", {
        "pref:text/coding": list(PREFERENCE_ROWS),
        "pref:text/creative-writing": literary,
        "bench:speed": [BenchmarkEntry("gemini-3-flash", "Gemini 3 Flash", "Google", 180.0)],
        "bench:reasoning-share/legal": [BenchmarkEntry("gemini-3-flash", "Gemini 3 Flash", "Google", 0.5)],
        "bench:reasoning-share/economics": [BenchmarkEntry("gemini-3-flash", "Gemini 3 Flash", "Google", 0.3),
                                            BenchmarkEntry("claude-opus-4-7", "Claude Opus 4.7", "Anthropic", 0.995)],
    })


@pytest.mark.parametrize("openrouter_id, name", [
    ("anthropic/claude-opus-4.7", "claude-opus-4-7-high"),
    ("anthropic/claude-opus-4.7", "claude-opus-4-7-20251101"),
    ("anthropic/claude-opus-4.7", "claude-opus-4-7-thinking-32k"),
    ("google/gemini-3-flash-preview", "gemini-3-flash"),
    ("openai/gpt-5.2:batch", "gpt-5-2-2025-12-11"),
    ("openai/gpt-5.2", "gpt-5.2-text"),
    ("x-ai/grok-4.1", "grok-4-1-beta2"),
    ("deepseek/deepseek-v4", "DeepSeek V4"),
    ("anthropic/claude-haiku-4.5", "claude-4-5-haiku-reasoning"),
    ("anthropic/claude-haiku-4.5", "Claude 4.5 Haiku (Reasoning)"),
    ("openai/gpt-6-astra", "gpt-6-astra-xhigh"),
    ("openai/gpt-5.2", "gpt-5-2-non-reasoning"),
])
def test_canonical_names_match(openrouter_id, name):
    assert canonical(openrouter_id) == canonical(name)


def test_canonical_keeps_different_models_apart():
    assert canonical("claude-opus-4.7") != canonical("claude-opus-4.6")
    assert canonical("gemini-3-flash") != canonical("gemini-3-pro")
    assert canonical("claude-sonnet-4-5") != canonical("claude-haiku-4-5")


def test_find_prefers_votes_then_shortest_key():
    assert snapshot().find("pref:text/coding", "anthropic/claude-opus-4.7").display_name == "claude-opus-4-7"
    variants = [BenchmarkEntry("gpt-6-astra-xhigh", "GPT-6 Astra (xhigh)", "OpenAI", 54.0),
                BenchmarkEntry("gpt-6-astra", "GPT-6 Astra (max)", "OpenAI", 55.0)]
    assert find("openai/gpt-6-astra", variants).model_key == "gpt-6-astra"
    assert snapshot().find("pref:text/coding", "meta/llama-9") is None


def test_quality_is_share_between_worst_and_best():
    shot = snapshot()
    assert shot.quality("pref:text/coding", "anthropic/claude-opus-4.7") == pytest.approx((1520.5 - 1300) / (1540.9 - 1300))
    assert shot.quality("pref:text/coding", "openai/gpt-5.2") == pytest.approx(0.0)
    assert shot.quality("pref:text/coding", "meta/llama-9") is None


def test_snapshot_survives_save_and_load(tmp_path):
    shot = snapshot()
    path = str(tmp_path / "benchmarks.json")
    shot.save(path)
    loaded = BenchmarkSnapshot.load(path)
    assert loaded.fetched_at == shot.fetched_at
    assert loaded.entries == shot.entries
    assert len(loaded.top(1).entries["pref:text/coding"]) == 1


def test_slug():
    assert slug("Finance/Investing") == "finance-investing"
    assert slug("Software Engineering (SWE)") == "software-engineering-swe"


def test_profiles_are_named_by_kind_of_measurement_and_have_full_dimension():
    assert all(key.startswith(("pref:", "bench:")) for key in benchmark_prior.PROFILES)
    assert len(benchmark_prior.PROFILES) == 81
    for tasks in benchmark_prior.PROFILES.values():
        assert all(len(item.feature_vector()) == Settings.full_dim() for item in tasks)


def test_prior_orders_models_by_series():
    """Модель сильнее в коде, чем в прозе, и прайор это воспроизводит; у другой наоборот."""
    Settings.task_mean = None
    shot = snapshot()
    code = benchmark_prior.PROFILES["pref:text/coding"][0].feature_vector()
    literary = benchmark_prior.PROFILES["pref:text/creative-writing"][0].feature_vector()

    claude = benchmark_prior.vector(shot, "anthropic/claude-opus-4.7")
    gemini = benchmark_prior.vector(shot, "google/gemini-3-flash-preview")
    assert claude is not None and gemini is not None
    assert float(code @ claude) > float(literary @ claude)
    assert float(code @ gemini) < float(literary @ gemini)
    assert benchmark_prior.vector(shot, "meta/llama-9") is None

    # Числа общие с C#-тестом. Прогноз чуть ниже качества в серии (0.9153): добавка к диагонали
    # системы Грама укорачивает вектор
    assert float(code @ claude) == pytest.approx(0.660785496860506, abs=1e-9)
    assert float(literary @ claude) == pytest.approx(0.10498806584289641, abs=1e-9)
    assert float(np.linalg.norm(claude)) == pytest.approx(0.7238061464221811, abs=1e-9)
    assert float(code @ gemini) == pytest.approx(0.41437720973970227, abs=1e-9)
    assert float(literary @ gemini) == pytest.approx(0.76953302847159, abs=1e-9)


def test_speed_and_cost_priors():
    shot = snapshot()
    assert benchmark_prior.tokens_per_second(shot, "google/gemini-3-flash-preview") == pytest.approx(180.0)
    assert benchmark_prior.tokens_per_second(shot, "anthropic/claude-opus-4.7") is None
    # Средняя доля рассуждений 0.4: задача дороже прайса ответа в 1/(1-0.4) раза
    assert benchmark_prior.cost_ratio(shot, "google/gemini-3-flash-preview") == pytest.approx(1 / 0.6)
    # Доля 0.995 дала бы двухсоткратную цену; поправка упирается в потолок
    assert benchmark_prior.cost_ratio(shot, "anthropic/claude-opus-4.7") == pytest.approx(benchmark_prior.MAX_COST_RATIO)
    assert benchmark_prior.cost_ratio(shot, "meta/llama-9") is None


def test_catalog_element_uses_benchmarks():
    model = catalog.ModelInfo(id="google/gemini-3-flash-preview", title="Gemini 3 Flash",
                              dollars_per_million_input=0.3, dollars_per_million_output=2.5,
                              context_tokens=1_000_000, max_answer_tokens=8000,
                              capabilities=catalog.Capability.NONE)
    plain = catalog.create_element(model)
    primed = catalog.create_element(model, benchmarks=snapshot())
    assert plain.tps == pytest.approx(catalog.DEFAULT_TOKENS_PER_SECOND)
    assert primed.tps == pytest.approx(180.0)
    assert primed.cost_ratio == pytest.approx(1 / 0.6)
    assert np.allclose(primed.ideal_match_vector, benchmark_prior.vector(snapshot(), model.id))


def test_known_series_and_field_quality():
    """Уровень поля: средняя доля качества по строкам известных серий; серии скорости и цены в него
    не входят. Снимок под другие имена серий профилям неизвестен: приора по нему нет ни у кого."""
    shot = snapshot()
    assert benchmark_prior.known_series(shot) == 2
    # Число общее с C#-тестом: четыре доли серии кода (1, 220.5/240.9, 100/240.9, 0) и две серии прозы (0, 1)
    assert benchmark_prior.field_quality(shot) == pytest.approx((1 + 220.5 / 240.9 + 100 / 240.9 + 0 + 0 + 1) / 6)

    foreign = BenchmarkSnapshot(shot.fetched_at, {"old:text/coding": list(PREFERENCE_ROWS)})
    assert benchmark_prior.known_series(foreign) == 0
    assert benchmark_prior.field_quality(foreign) is None
    assert benchmark_prior.vector(foreign, "anthropic/claude-opus-4.7") is None


def test_features_without_recognition_take_the_typical_task():
    """Без распознавания задание неизвестно, и признаки берут спецификацию типовой задачи, а не
    пустую: пустая проецировала прогноз на случайное направление (docs/research/prior-holdout.md)."""
    from fai_router.services import InputFeaturesService

    features = InputFeaturesService.get_features("привет")
    typical = benchmark_prior.typical_task().input_specifications

    assert features.input_specifications.style_type == typical.style_type
    assert features.input_specifications.symbol_length == typical.symbol_length
    assert features.input_len == pytest.approx(len("привет") / InputFeaturesService.EST_SYMBOL_PER_TOKEN)
    # Объект свой у каждого вызова: правка признаков одного хода не трогает следующий
    assert features.input_specifications is not InputFeaturesService.get_features("привет").input_specifications


# Каталог поставщика, кандидаты из него, снимок и имена: те же проверки, что в CatalogTests.cs


def test_openrouter_catalog_survives_broken_items():
    """Битая запись пропускается, а не роняет весь каталог; цена без данных неизвестна, а не бесплатна."""
    text = json.dumps({"data": [
        {"id": "ok/model", "name": "Ok", "pricing": {"prompt": "0.000001", "completion": 0.000002, "web_search": "0.01"},
         "context_length": 128000.5, "top_provider": {"max_completion_tokens": "4096"},
         "architecture": {"input_modalities": None}, "supported_parameters": ["tools", 5]},
        {"name": "без идентификатора"},
        {"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "auto/model", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "noprice/model", "pricing": None, "architecture": "строка"},
        "мусор", None,
    ]}, ensure_ascii=False)
    models = {model.id: model for model in catalog.parse(text)}
    assert list(models) == ["ok/model", "free/model", "auto/model", "noprice/model"]
    ok = models["ok/model"]
    assert ok.dollars_per_million_input == pytest.approx(1) and ok.dollars_per_million_output == pytest.approx(2)
    assert (ok.context_tokens, ok.max_answer_tokens) == (128000, 4096)
    assert ok.capabilities == (catalog.Capability.CODE | catalog.Capability.FORMULAS | catalog.Capability.TOOLS
                               | catalog.Capability.WEB_SEARCH)
    assert models["free/model"].dollars_per_million_input == 0
    assert models["auto/model"].dollars_per_million_input < 0
    assert models["noprice/model"].dollars_per_million_output < 0
    assert "noprice/model" not in catalog.select("all", models)[0]


def test_fractalrouter_catalog_survives_broken_items():
    text = json.dumps([
        {"id": "a/model", "pricing": {"prompt_rub_per_1m": "12,5", "completion_rub_per_1m": 50},
         "modalities": ["text", "image"], "max_completion_tokens": 2048.7},
        {"id": "b/model", "pricing": "нет", "modalities": None},
        {"id": 17},
    ], ensure_ascii=False)
    models = {model.id: model for model in catalog.parse_fractalrouter(text)}
    assert models["a/model"].dollars_per_million_input == pytest.approx(12.5)
    assert models["a/model"].max_answer_tokens == 2048
    assert models["a/model"].capabilities & catalog.Capability.VISION
    assert models["b/model"].dollars_per_million_input < 0
    assert len(models) == 3


def test_candidates_keep_catalog_capabilities_and_skip_repeats():
    """Цена из prices поверх каталога не стирает возможностей модели; модель только из prices
    считается способной на все; повтор в списке не роняет роутер."""
    capability = catalog.Capability
    known = {"a/model": catalog.ModelInfo("a/model", "A", 1, 2, 100_000, 4_000, capability.CODE | capability.VISION),
             "auto/model": catalog.ModelInfo("auto/model", "Auto", -1, -1, 0, 0, capability.CODE)}
    prices = {"a/model": (3, 4), "x/model": (5, 6)}
    candidates = catalog.create_candidates(["a/model", "a/model", "x/model", "auto/model"], known, prices, task_mean=None)
    assert [c.name for c in candidates] == ["a/model", "x/model", "auto/model"]
    assert candidates[0].capabilities == capability.CODE | capability.VISION
    assert candidates[0].dpmt_inp == 3 and candidates[0].context_window == 100_000
    assert candidates[1].capabilities == capability.ALL
    assert not candidates[2].has_known_price


def test_unrated_model_starts_from_the_field_level():
    """Модель без рейтингов получает уровень поля, а не случайный вектор; скорость вне серии это
    нижняя граница серии, а не 50."""
    from fai_router.benchmarks import default_snapshot

    shot = default_snapshot()
    unknown = catalog.ModelInfo("vendor/never-rated-model-x", "X", 1, 2, 0, 0, catalog.Capability.ALL)
    first = catalog.create_element(unknown, None, shot, None)
    second = catalog.create_element(unknown, None, shot, None)
    assert np.array_equal(first.ideal_match_vector, second.ideal_match_vector)
    assert first.prior_experience == 0
    assert first.tps == pytest.approx(min(row.score for row in shot.entries["bench:speed"]))
    assert np.allclose(first.ideal_match_vector,
                       benchmark_prior.uniform(benchmark_prior.field_quality(shot), None))


def test_uniform_is_linear_and_on_the_rated_scale():
    """uniform считается один раз для единицы: вектор линеен по качеству, а безрейтинговая модель с
    уровнем поля не обгоняет сильнейшие рейтинговые."""
    from fai_router.benchmarks import default_snapshot

    unit = benchmark_prior.uniform(1.0, None)
    assert np.allclose(benchmark_prior.uniform(0.4, None), unit * 0.4)
    shot = default_snapshot()
    typical = benchmark_prior.typical_task().feature_vector()
    field = benchmark_prior.field_quality(shot)
    rated = [benchmark_prior.vector(shot, model_id, None) for model_id in catalog.popular_models()]
    forecasts = sorted(float(typical @ vector) for vector in rated if vector is not None)
    assert float(typical @ benchmark_prior.uniform(field, None)) < forecasts[-1]


@pytest.mark.parametrize("product, base", [
    ("qwen/qwen3.8-max", "qwen/qwen3.8"),
    ("openai/gpt-5.1-codex-max", "openai/gpt-5.1-codex"),
    ("moonshotai/kimi-k2-thinking", "moonshotai/kimi-k2"),
])
def test_product_suffixes_stay_apart(product, base):
    """Хвосты max и thinking у одних поставщиков означают другой продукт: точное имя их хранит,
    семейство срезает."""
    assert canonical(product) != canonical(base)
    assert canonical(product, relaxed=True) == canonical(base, relaxed=True)


def test_claude_thinking_is_the_same_model():
    assert canonical("anthropic/claude-opus-4.7") == canonical("claude-opus-4-7-thinking-32k")
    assert canonical("anthropic/claude-opus-4.1") == canonical("claude-opus-4-1-20250805-thinking-16k")


def test_find_prefers_exact_names_then_family():
    thinking = BenchmarkEntry("kimi-k2-thinking", "Kimi K2 Thinking", "Moonshot", 50)
    plain = BenchmarkEntry("kimi-k2", "Kimi K2", "Moonshot", 40)
    assert find("moonshotai/kimi-k2-thinking", [plain, thinking]) is thinking
    assert find("moonshotai/kimi-k2", [plain, thinking]) is plain
    assert find("moonshotai/kimi-k2", [thinking]) is thinking
    shot = BenchmarkSnapshot("", {"pref:text/overall": [plain, thinking]})
    assert shot.find("pref:text/overall", "moonshotai/kimi-k2-thinking") is thinking
    assert shot.find("pref:text/overall", "moonshotai/kimi-k2") is plain


def test_indexed_find_matches_the_scan():
    from fai_router.benchmarks import default_snapshot

    shot = default_snapshot()
    names = list(dict.fromkeys("vendor/" + row.model_key for rows in shot.entries.values() for row in rows))[:300]
    for key, rows in list(shot.entries.items())[:20]:
        for name in names:
            assert shot.find(key, name) is find(name, rows)


def test_top_keeps_full_series_bounds(tmp_path):
    """Обрезка снимка хранит границы полной серии: доля качества по обрезку та же, что по полной
    серии, уровень поля по средней полной серии, а формат обратно совместим."""
    full = BenchmarkSnapshot("2026-10-01T00:00:00Z", {"pref:text/overall": [
        BenchmarkEntry("a", "a", "", 1500), BenchmarkEntry("b", "b", "", 1490),
        BenchmarkEntry("c", "c", "", 1200), BenchmarkEntry("d", "d", "", 1100)]})
    cut = full.top(2)
    assert len(cut.entries["pref:text/overall"]) == 2
    assert cut.quality("pref:text/overall", "vendor/b") == full.quality("pref:text/overall", "vendor/b")
    assert cut.quality("pref:text/overall", "vendor/b") == pytest.approx(0.975)
    assert cut.bounds["pref:text/overall"].count == 4
    assert cut.mean_share("pref:text/overall")[0] == pytest.approx((1322.5 - 1100) / 400)

    plain_path, cut_path = str(tmp_path / "full.json"), str(tmp_path / "cut.json")
    full.save(plain_path)
    cut.save(cut_path)
    with open(plain_path, encoding="utf-8") as file:
        assert "bounds" not in json.load(file)
    loaded = BenchmarkSnapshot.load(cut_path)
    assert loaded.quality("pref:text/overall", "vendor/b") == pytest.approx(0.975)


def test_snapshot_age():
    from datetime import datetime, timedelta, timezone

    old = BenchmarkSnapshot((datetime.now(timezone.utc) - timedelta(days=200)).isoformat(), {})
    assert old.age_days == pytest.approx(200, abs=0.01)
    assert BenchmarkSnapshot("не дата", {}).age_days is None
