"""Снимок внешних замеров: сопоставление имен, качество в серии, начальный вектор, скорость и цена.

Ожидаемые числа общие с C#-тестом (Mas.Core.Tests/RouterBenchmarkTests.cs): расхождение версий
видно по расхождению чисел."""

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
