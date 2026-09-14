"""Рейтинги Artificial Analysis (artificialanalysis.ai): отраслевые индексы, фактология, бизнес-
функции агентной работы, документная работа по отраслям, скорость и цена задачи.

Официальный API требует ключа, а страницы отдают массив моделей со всеми оценками внутри кусков
RSC, как и арена. Берутся восемь страниц: шесть индексов (около 24 моделей каждый), страница
AutomationBench (30 передовых моделей со всеми разбивками) и общая таблица (646 моделей, основные
оценки, скорость и цены). Серии, где меньше значит лучше, здесь не берутся: вместо доли выдумки
берется доля ответов без выдумки (omniscienceNonHallucination)."""

from __future__ import annotations

import re
from typing import Any

from fai_router.benchmarks import BenchmarkEntry, get, json_value_at, rsc_payload, slug

AA_URL = "https://artificialanalysis.ai/{path}"

# Отраслевые индексы: адрес models/capabilities/<индекс>
INDICES = ("finance-and-accounting", "strategy-and-ops", "legal", "healthcare-and-medical",
           "engineering", "economics")

EVALUATION_PATH = "evaluations/automationbench-aa"
LEADERBOARD_PATH = "leaderboards/models"

# Серии общей таблицы: ключ снимка и поле модели
LEADERBOARD_SERIES = {
    "aa:intelligence": "intelligenceIndex",
    "aa:non-hallucination": "omniscienceNonHallucination",
    "aa:ifbench": "ifbench",
    "aa:lcr": "lcr",
    "aa:tau2": "tau2",
    "aa:hle": "hle",
    "aa:gpqa": "gpqa",
    "aa:gdpval": "gdpvalNormalized",
    "aa:terminalbench-hard": "terminalbenchHard",
    "aa:scicode": "scicode",
    "aa:speed": "medianOutputTokensPerSecond",
}

_MODEL_ARRAY = re.compile(r'"(?:initialModels|models)":\[')
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")

Series = dict[str, list[BenchmarkEntry]]


def models_of(html: str) -> list[dict[str, Any]]:
    """Массив моделей страницы. На странице бывает несколько массивов (меню, выборка для графика),
    берется тот, у записей которого больше всего полей."""
    text = rsc_payload(html)
    best: list[dict[str, Any]] = []
    for match in _MODEL_ARRAY.finditer(text):
        rows = json_value_at(text, match.end() - 1)
        if rows and isinstance(rows[0], dict) and (not best or len(rows[0]) > len(best[0])):
            best = rows
    if not best:
        raise ValueError("На странице нет массива моделей: адрес не тот или разметка изменилась.")
    return best


def index_series(index: str, models: list[dict[str, Any]]) -> Series:
    """Индекс, его части, цена задачи и доля рассуждений в ней. Доля рассуждений дает начальную
    поправку цены: прайс не знает, сколько модель потратит на размышление."""
    series: Series = {}
    for model in models:
        _add(series, f"aa:index/{index}", model, model.get("weightedIndex"))
        # Части индекса названы ключами JSON в camelCase: nonHallucination становится non-hallucination
        for part, value in (model.get("subScores") or {}).items():
            name = slug(_CAMEL_BOUNDARY.sub(r"\1-\2", part))
            _add(series, f"aa:index/{index}/{name}", model, value)
        cost = model.get("costPerTask") or {}
        total, reasoning = cost.get("total"), cost.get("reasoning")
        _add(series, f"aa:cost-per-task/{index}", model, total)
        if _number(total) and _number(reasoning) and total > 0:
            _add(series, f"aa:reasoning-share/{index}", model, reasoning / total)
    return series


def evaluation_series(models: list[dict[str, Any]]) -> Series:
    """Разбивки передовых моделей: фактология по областям знаний и языкам программирования,
    бизнес-функции и сервисы AutomationBench, отрасли GDP.pdf, аналитика и оформление Briefcase,
    практика права Harvey."""
    series: Series = {}
    for model in models:
        omniscience = model.get("omniscienceBreakdown") or {}
        for name, value in (omniscience.get("byDomain") or {}).items():
            _add(series, f"aa:omniscience/{slug(name)}", model, value)
        for name, value in (omniscience.get("bySweLanguage") or {}).items():
            _add(series, f"aa:swe-language/{slug(name)}", model, value)

        automation = model.get("automationBenchBreakdown") or {}
        for name, value in (automation.get("byDomain") or {}).items():
            _add(series, f"aa:automation/{slug(name)}", model, (value or {}).get("completion"))
        for name, value in (automation.get("byApp") or {}).items():
            _add(series, f"aa:automation-app/{slug(name)}", model, (value or {}).get("completion"))

        for name, value in ((model.get("gdpPdfBreakdown") or {}).get("byDomain") or {}).items():
            _add(series, f"aa:gdp-pdf/{slug(name)}", model, value)

        briefcase = model.get("briefcaseBreakdown") or {}
        _add(series, "aa:briefcase/analytical", model, (briefcase.get("analyticalQuality") or {}).get("elo"))
        _add(series, "aa:briefcase/presentation", model, (briefcase.get("presentation") or {}).get("elo"))
        _add(series, "aa:harvey", model, model.get("harveyLab"))
    return series


def leaderboard_series(models: list[dict[str, Any]]) -> Series:
    """Основные оценки, скорость и цены по общей таблице."""
    series: Series = {}
    for model in models:
        for key, field in LEADERBOARD_SERIES.items():
            _add(series, key, model, model.get(field))
    return series


def fetch_all(timeout: float = 60.0) -> Series:
    series: Series = {}
    for index in INDICES:
        series.update(index_series(index, models_of(get(AA_URL.format(path=f"models/capabilities/{index}"), timeout))))
    series.update(evaluation_series(models_of(get(AA_URL.format(path=EVALUATION_PATH), timeout))))
    series.update(leaderboard_series(models_of(get(AA_URL.format(path=LEADERBOARD_PATH), timeout))))
    # Внутри серии порядок по убыванию оценки, как у арены: тогда урезанный снимок держит лучших
    return {key: sorted(rows, key=lambda row: -row.score) for key, rows in series.items()}


def _add(series: Series, key: str, model: dict[str, Any], value: Any) -> None:
    if not _number(value):
        return
    creator = model.get("creator") or {}
    series.setdefault(key, []).append(BenchmarkEntry(
        model_key=str(model.get("slug", "")),
        display_name=str(model.get("name") or model.get("shortName") or ""),
        organization=str(creator.get("name") or model.get("modelCreatorName") or ""),
        score=float(value),
    ))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
