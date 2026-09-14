"""Рейтинги: разбор арены и Artificial Analysis, сопоставление имен, качество, начальный вектор.

Фикстуры страниц и ожидаемые числа общие с C#-тестом (Mas.Core.Tests/RouterBenchmarkTests.cs):
расхождение версий видно по расхождению чисел."""

import json

import numpy as np
import pytest

from fai_router import analysis, arena, catalog
from fai_router.benchmarks import BenchmarkEntry, BenchmarkSnapshot, canonical, find, slug
from fai_router.settings import Settings
from fai_router.training import benchmark_prior

ARENA_ENTRIES = [
    {"rank": 1, "modelKey": "claude-opus-4-7-thinking", "modelDisplayName": "claude-opus-4-7-high",
     "modelOrganization": "Anthropic", "rating": 1540.9, "votes": 24222},
    {"rank": 2, "modelKey": "claude-opus-4-7-text", "modelDisplayName": "claude-opus-4-7",
     "modelOrganization": "Anthropic", "rating": 1520.5, "votes": 30000},
    {"rank": 3, "modelKey": "gemini-3-flash-text", "modelDisplayName": "gemini-3-flash",
     "modelOrganization": "Google", "rating": 1400.0, "votes": 9000},
    {"rank": 4, "modelKey": "gpt-5-2-2025-12-11-text", "modelDisplayName": "gpt-5.2-2025-12-11",
     "modelOrganization": "OpenAI", "rating": 1300.0, "votes": 100},
]

# Запись модели AA в том виде, в каком ее отдает страница оценки AutomationBench
AA_MODEL = {
    "slug": "claude-4-5-haiku-reasoning", "name": "Claude 4.5 Haiku (Reasoning)", "shortName": "Claude 4.5 Haiku",
    "creator": {"name": "Anthropic"},
    "weightedIndex": 41.5,
    "subScores": {"knowledge": 40.0, "nonHallucination": 30.0},
    "costPerTask": {"total": 2.0, "reasoning": 0.5, "answer": 0.7},
    "omniscienceBreakdown": {"byDomain": {"Business": 29.3, "Software Engineering (SWE)": 86.2},
                             "bySweLanguage": {"Python": 91.0, "HTML": 84.0}},
    "automationBenchBreakdown": {"byDomain": {"marketing": {"completion": 0.9}},
                                 "byApp": {"google_sheets": {"completion": 0.8}}},
    "gdpPdfBreakdown": {"byDomain": {"Finance/Investing": 0.82}},
    "briefcaseBreakdown": {"analyticalQuality": {"elo": 1958.5}, "presentation": {"elo": 1472.7}},
    "harveyLab": 0.93,
    "omniscienceNonHallucination": 0.41, "ifbench": None, "medianOutputTokensPerSecond": 120.0,
}


def page(payload: str) -> str:
    """Страница в том виде, в каком ее отдает сервер: JSON внутри строкового литерала JS."""
    literal = json.dumps(payload)[1:-1]
    return ('<html><script>self.__next_f.push([1,"3:I[1,2]\\n"])</script>'
            '<script>self.__next_f.push([1,"' + literal + '"])</script></html>')


def arena_page(key: str = "text-coding-style_control") -> str:
    return page('{"id":"leaderboard-sets/public/leaderboards/' + key + '/leaderboard-snapshots/latest","entries":'
                + json.dumps(ARENA_ENTRIES) + ',"extra":[1,2]}')


def aa_page() -> str:
    # Короткий массив на странице есть всегда (меню моделей); брать надо массив с полными записями
    return page('{"models":[{"slug":"menu"}],"initialModels":' + json.dumps([AA_MODEL]) + '}')


def snapshot() -> BenchmarkSnapshot:
    rows = [BenchmarkEntry(e["modelKey"], e["modelDisplayName"], e["modelOrganization"], e["rating"], e["votes"])
            for e in ARENA_ENTRIES]
    literary = [BenchmarkEntry("claude-opus-4-7-text", "claude-opus-4-7", "Anthropic", 1400.0, 100),
                BenchmarkEntry("gemini-3-flash-text", "gemini-3-flash", "Google", 1500.0, 100)]
    return BenchmarkSnapshot("2026-09-14T00:00:00+00:00", {
        "arena:text/coding": rows,
        "arena:text/creative-writing": literary,
        "aa:speed": [BenchmarkEntry("gemini-3-flash", "Gemini 3 Flash", "Google", 180.0)],
        "aa:reasoning-share/legal": [BenchmarkEntry("gemini-3-flash", "Gemini 3 Flash", "Google", 0.5)],
        "aa:reasoning-share/economics": [BenchmarkEntry("gemini-3-flash", "Gemini 3 Flash", "Google", 0.3),
                                         BenchmarkEntry("claude-opus-4-7", "Claude Opus 4.7", "Anthropic", 0.995)],
    })


def test_arena_page_parses():
    key, rows = arena.parse(arena_page())
    assert key == "text-coding-style_control"
    assert [row.display_name for row in rows] == ["claude-opus-4-7-high", "claude-opus-4-7", "gemini-3-flash",
                                                  "gpt-5.2-2025-12-11"]
    assert rows[0].score == pytest.approx(1540.9)
    assert rows[0].votes == 24222


def test_page_without_leaderboard_raises():
    with pytest.raises(ValueError):
        arena.parse("<html>nothing here</html>")
    with pytest.raises(ValueError):
        analysis.models_of("<html>nothing here</html>")


def test_aa_page_takes_the_full_model_array():
    models = analysis.models_of(aa_page())
    assert [model["slug"] for model in models] == ["claude-4-5-haiku-reasoning"]


def test_aa_series():
    models = analysis.models_of(aa_page())
    index = analysis.index_series("legal", models)
    assert index["aa:index/legal"][0].score == pytest.approx(41.5)
    assert index["aa:index/legal/non-hallucination"][0].score == pytest.approx(30.0)
    assert index["aa:cost-per-task/legal"][0].score == pytest.approx(2.0)
    assert index["aa:reasoning-share/legal"][0].score == pytest.approx(0.25)

    evaluation = analysis.evaluation_series(models)
    assert set(evaluation) == {
        "aa:omniscience/business", "aa:omniscience/software-engineering-swe",
        "aa:swe-language/python", "aa:swe-language/html",
        "aa:automation/marketing", "aa:automation-app/google-sheets", "aa:gdp-pdf/finance-investing",
        "aa:briefcase/analytical", "aa:briefcase/presentation", "aa:harvey",
    }
    entry = evaluation["aa:automation/marketing"][0]
    assert (entry.model_key, entry.display_name, entry.organization, entry.score) == \
        ("claude-4-5-haiku-reasoning", "Claude 4.5 Haiku (Reasoning)", "Anthropic", 0.9)

    table = analysis.leaderboard_series(models)
    assert set(table) == {"aa:non-hallucination", "aa:speed"}


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
    assert snapshot().find("arena:text/coding", "anthropic/claude-opus-4.7").display_name == "claude-opus-4-7"
    variants = [BenchmarkEntry("gpt-6-astra-xhigh", "GPT-6 Astra (xhigh)", "OpenAI", 54.0),
                BenchmarkEntry("gpt-6-astra", "GPT-6 Astra (max)", "OpenAI", 55.0)]
    assert find("openai/gpt-6-astra", variants).model_key == "gpt-6-astra"
    assert snapshot().find("arena:text/coding", "meta/llama-9") is None


def test_quality_is_share_between_worst_and_best():
    shot = snapshot()
    assert shot.quality("arena:text/coding", "anthropic/claude-opus-4.7") == pytest.approx((1520.5 - 1300) / (1540.9 - 1300))
    assert shot.quality("arena:text/coding", "openai/gpt-5.2") == pytest.approx(0.0)
    assert shot.quality("arena:text/coding", "meta/llama-9") is None


def test_snapshot_survives_save_and_load(tmp_path):
    shot = snapshot()
    path = str(tmp_path / "benchmarks.json")
    shot.save(path)
    loaded = BenchmarkSnapshot.load(path)
    assert loaded.fetched_at == shot.fetched_at
    assert loaded.entries == shot.entries
    assert len(loaded.top(1).entries["arena:text/coding"]) == 1


def test_slug():
    assert slug("Finance/Investing") == "finance-investing"
    assert slug("Software Engineering (SWE)") == "software-engineering-swe"


def test_profiles_cover_every_arena_category_and_have_full_dimension():
    assert {category.key for category in arena.CATEGORIES} <= set(benchmark_prior.PROFILES)
    assert all(key.startswith(("arena:", "aa:")) for key in benchmark_prior.PROFILES)
    for tasks in benchmark_prior.PROFILES.values():
        assert all(len(item.feature_vector()) == Settings.full_dim() for item in tasks)


def test_prior_orders_models_by_category():
    """Модель сильнее в коде, чем в прозе, и прайор это воспроизводит; у другой наоборот."""
    Settings.task_mean = None
    shot = snapshot()
    code = benchmark_prior.PROFILES["arena:text/coding"][0].feature_vector()
    literary = benchmark_prior.PROFILES["arena:text/creative-writing"][0].feature_vector()

    claude = benchmark_prior.vector(shot, "anthropic/claude-opus-4.7")
    gemini = benchmark_prior.vector(shot, "google/gemini-3-flash-preview")
    assert claude is not None and gemini is not None
    assert float(code @ claude) > float(literary @ claude)
    assert float(code @ gemini) < float(literary @ gemini)
    assert benchmark_prior.vector(shot, "meta/llama-9") is None

    # Числа общие с C#-тестом. Прогноз чуть ниже качества в серии (0.9153): добавка к диагонали
    # системы Грама укорачивает вектор
    assert float(code @ claude) == pytest.approx(0.9140351676590679, abs=1e-9)
    assert float(literary @ claude) == pytest.approx(0.0006869535025922358, abs=1e-9)
    assert float(np.linalg.norm(claude)) == pytest.approx(1.082441231294556, abs=1e-9)
    assert float(code @ gemini) == pytest.approx(0.41527892909152475, abs=1e-9)
    assert float(literary @ gemini) == pytest.approx(0.9989105090218988, abs=1e-9)


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
                              capabilities=catalog.Capability.NONE, intelligence_index=0.0,
                              coding_index=0.0, agentic_index=0.0)
    plain = catalog.create_element(model)
    primed = catalog.create_element(model, benchmarks=snapshot())
    assert plain.tps == pytest.approx(catalog.DEFAULT_TOKENS_PER_SECOND)
    assert primed.tps == pytest.approx(180.0)
    assert primed.cost_ratio == pytest.approx(1 / 0.6)
    assert np.allclose(primed.ideal_match_vector, benchmark_prior.vector(snapshot(), model.id))
