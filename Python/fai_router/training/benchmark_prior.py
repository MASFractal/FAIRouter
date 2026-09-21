"""Начальный вектор кандидата из рейтингов арены и Artificial Analysis.

У рейтингов есть оценки моделей по сериям (категории арены, отраслевые индексы, фактология по
областям), а у роутера есть механизм начальных весов из замеров по типам задач
(from_measurements). Здесь серии превращаются в типы задач: каждой серии соответствует профиль
признаков, а оценка модели в ней становится качеством на этом профиле.

Профили лежат в общем файле data/benchmark_profiles.json: версия на C# читает тот же файл, и
расходиться профилям негде. Скорость и доля рассуждений в цене задачи дают начальные скорость и
поправку цены кандидата, пока своих замеров нет."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fai_router.analysis import INDICES
from fai_router.specifications import Specifications
from fai_router.tracking import InputFeatures
from fai_router.training.quality_prior import from_measurements

if TYPE_CHECKING:
    from fai_router.benchmarks import BenchmarkSnapshot

PROFILES_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_profiles.json"

# Поправка цены из доли рассуждений: во сколько раз задача дороже прайса ответа. Верх нужен
# потому, что доля рассуждений бывает под 0,99, а один такой замер не повод считать модель
# стократно дороже
MAX_COST_RATIO = 5.0


def _load() -> dict[str, Any]:
    with open(PROFILES_PATH, encoding="utf-8") as file:
        return json.load(file)


_SOURCE = _load()


def task(item: dict[str, Any]) -> InputFeatures:
    """Задача по записи профиля: base.spec, затем шаблон, затем spec профиля."""
    base = _SOURCE["base"]
    spec = dict(base["spec"])
    if "template" in item:
        spec.update(_SOURCE["templates"][item["template"]])
    spec.update(item.get("spec", {}))
    return InputFeatures(
        input_len=float(item.get("input_len", base["input_len"])),
        len_answer=float(item.get("len_answer", base["len_answer"])),
        input_specifications=Specifications.from_dict(spec),
        turn_count=int(item.get("turn_count", base["turn_count"])),
    )


def typical_task() -> InputFeatures:
    """Опорная задача: обычный развернутый ответ пользователю."""
    return task({})


# Профили по ключам серий, в порядке файла
PROFILES: dict[str, list[InputFeatures]] = {
    key: [task(item) for item in items] for key, items in _SOURCE["profiles"].items()
}


def measurements(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> list[tuple[np.ndarray, float]]:
    """Пары «вектор задачи, качество» для from_measurements по всем сериям, где модель есть."""
    pairs = []
    for key, tasks in PROFILES.items():
        quality = snapshot.quality(key, openrouter_id)
        if quality is None:
            continue
        pairs.extend((item.feature_vector(), quality) for item in tasks)
    return pairs


def vector(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> np.ndarray | None:
    """Начальный вектор кандидата по рейтингам; None, если модели нет ни в одной серии."""
    pairs = measurements(snapshot, openrouter_id)
    return from_measurements(pairs) if pairs else None


def uniform(quality: float) -> np.ndarray:
    """Начальный вектор модели без рейтингов: качество одинаково во всех сериях.

    Та же подгонка по тем же профилям, что у vector(), поэтому шкала общая с моделями из рейтингов:
    вектор по одной опорной задаче сжимается иначе, и безрейтинговая модель обгоняла рейтинговые."""
    return from_measurements([(item.feature_vector(), 1.0) for tasks in PROFILES.values() for item in tasks]) * quality


def tokens_per_second(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> float | None:
    """Скорость модели по замеру Artificial Analysis, токенов в секунду; None, если замера нет."""
    speed = snapshot.value("aa:speed", openrouter_id)
    return speed if speed is not None and speed > 0 else None


def cost_ratio(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> float | None:
    """Начальная поправка цены: задача обходится дороже прайса ответа на долю рассуждений.
    Среднее по отраслевым индексам, где модель есть; None, если ее нет ни в одном."""
    shares = [share for share in (snapshot.value(f"aa:reasoning-share/{index}", openrouter_id) for index in INDICES)
              if share is not None]
    if not shares:
        return None
    share = sum(shares) / len(shares)
    return min(max(1.0 / max(1.0 - share, 1e-9), 1.0), MAX_COST_RATIO)
