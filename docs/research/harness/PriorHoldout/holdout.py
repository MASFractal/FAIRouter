"""Офлайн-замер выбора модели: выбросить серию рейтингов и предсказать ее.

Для каждой серии снимка, у которой есть профиль задачи, приор кандидатов строится по всем сериям,
КРОМЕ нее, а роутер выбирает модель под задачу этой серии. Истинное качество выбранной модели берется
из выброшенной серии. Так проверяется весь путь выбора: приор по рейтингам, признаки задачи и метрика
R, на сотнях задач и без единого обращения к моделям.

Кандидатские группы двух видов: весь список популярных моделей, у которых есть оценка в серии
(режим ползунков хоста), и случайные пятерки из них (режим «Авто», где хост держит несколько
измеренных моделей). Цена модели берется из списка популярных моделей.

Запуск из каталога Python: py -3 ../docs/research/harness/PriorHoldout/holdout.py [вариант...]
"""

from __future__ import annotations

import json
import os
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "Python"
sys.path.insert(0, str(ROOT))

from fai_router import env  # noqa: E402
from fai_router.benchmarks import BenchmarkSnapshot, default_snapshot  # noqa: E402
from fai_router.routed_element import RoutedElement  # noqa: E402
from fai_router.settings import RouteWeights  # noqa: E402
from fai_router.training import benchmark_prior  # noqa: E402

POOLS_PER_SERIES = 40
POOL_SIZE = 5
MIN_MODELS = 6
DEFAULT_TPS = 50.0

PROFILES = {
    "качество": RouteWeights(0.8, 0.10, 0.10, 0.0),
    "баланс": RouteWeights(0.5, 0.25, 0.25, 0.0),
    "экономия": RouteWeights(0.3, 0.60, 0.10, 0.0),
}


def popular() -> dict[str, tuple[float, float]]:
    data = json.loads((ROOT / "fai_router" / "data" / "popular_models.json").read_text(encoding="utf-8"))
    return {m["id"]: (m["usd_per_million_input"], m["usd_per_million_output"]) for m in data["models"]}


def without(snapshot: BenchmarkSnapshot, key: str) -> BenchmarkSnapshot:
    return BenchmarkSnapshot(snapshot.fetched_at, {k: v for k, v in snapshot.entries.items() if k != key})


def element(model_id: str, price: tuple[float, float], train: BenchmarkSnapshot, full: BenchmarkSnapshot,
            field: float) -> tuple[RoutedElement, bool]:
    vector = benchmark_prior.vector(train, model_id)
    rated = vector is not None
    item = RoutedElement(
        name=model_id,
        tps=benchmark_prior.tokens_per_second(full, model_id) or DEFAULT_TPS,
        dpmt_inp=price[0],
        dpmt_outp=price[1],
        ideal_match_vector=vector if rated else benchmark_prior.uniform(field),
    )
    item.cost_ratio = benchmark_prior.cost_ratio(full, model_id) or 1.0
    return item, rated


def blended(price: tuple[float, float]) -> float:
    """Цена за миллион токенов хода: вход и выход в пропорции обычного ответа чата, один к трем."""
    return (price[0] + 3 * price[1]) / 4


def run(variant: str = "текущий") -> dict:
    snapshot = default_snapshot()
    prices = popular()
    rng = random.Random(20261008)
    rows: dict[str, list[tuple[float, float, float]]] = {}

    for key, tasks in benchmark_prior.PROFILES.items():
        truth = {m: snapshot.quality(key, m) for m in prices}
        present = [m for m, q in truth.items() if q is not None]
        if len(present) < MIN_MODELS:
            continue
        train = without(snapshot, key)
        field = benchmark_prior.field_quality(train) or 0.5
        built = {m: element(m, prices[m], train, snapshot, field)[0] for m in present}
        # Глобальный рейтинг без задачи: средняя доля модели по остальным сериям
        overall = {m: statistics.mean([q for _, q in benchmark_prior.measurements(train, m)] or [0.0]) for m in present}

        pools = [present] + [rng.sample(present, POOL_SIZE) for _ in range(POOLS_PER_SERIES)]
        for pool_index, pool in enumerate(pools):
            kind = "все" if pool_index == 0 else "пятерки"
            for task in tasks[:1]:
                best = max(truth[m] for m in pool)
                picks = {
                    "оракул": max(pool, key=lambda m: truth[m]),
                    "дешевая": min(pool, key=lambda m: blended(prices[m])),
                    "дорогая": max(pool, key=lambda m: blended(prices[m])),
                    "общий рейтинг": max(pool, key=lambda m: overall[m]),
                }
                for name, weights in PROFILES.items():
                    top = env.get_top_k(task, [built[m] for m in pool], 1, weights=weights)
                    picks[f"роутер, {name}"] = top[0][1].name
                for name, model in picks.items():
                    rows.setdefault(f"{kind} · {name}", []).append(
                        (truth[model], blended(prices[model]), best - truth[model]))
                # Случайный выбор: математическое ожидание по группе
                rows.setdefault(f"{kind} · случайная", []).append(
                    (statistics.mean(truth[m] for m in pool), statistics.mean(blended(prices[m]) for m in pool),
                     best - statistics.mean(truth[m] for m in pool)))

    return {k: (statistics.mean(r[0] for r in v), statistics.mean(r[1] for r in v), statistics.mean(r[2] for r in v), len(v))
            for k, v in rows.items()}


def score(variant: str, task, pool: list[RoutedElement], weights: RouteWeights) -> RoutedElement:
    """Победитель по варианту метрики. «текущий» это сама библиотека, остальные пробуются здесь."""
    if variant == "текущий":
        return env.get_top_k(task, pool, 1, weights=weights)[0][1]

    vector = task.feature_vector()
    q = [e.get_quality_score(vector) for e in pool]
    c = [math.log(e.get_cost(task) + env.COST_FLOOR) for e in pool]
    t = [math.log(e.get_time(task) + 2) for e in pool]

    if variant.startswith("порог"):
        # Разброс не меньше осмысленного: шум в прогнозе качества не раздувается до целой единицы
        fq, fc, ft = FLOORS[variant]
        q, c, t = floored(q, fq), floored(c, fc), floored(t, ft)
    best = max(range(len(pool)), key=lambda i: weights.WQ * q[i] - weights.WC * c[i] - weights.WT * t[i])
    return pool[best]


FLOORS = {
    "порог": (0.05, math.log(2), math.log(2)),
    "порог-мягкий": (0.02, math.log(1.5), math.log(1.5)),
    "порог-жесткий": (0.1, math.log(4), math.log(4)),
}


def floored(values: list[float], floor: float) -> list[float]:
    mean = statistics.fmean(values)
    deviation = max(statistics.pstdev(values), floor)
    return [(v - mean) / deviation for v in values]


def sweep(variant: str) -> dict[str, list[tuple[float, float, float]]]:
    """Кривая цена-качество по ползунку: WQ от 0 до 1, остаток поровну цене и времени."""
    snapshot = default_snapshot()
    prices = popular()
    curves: dict[str, list[tuple[float, float, float]]] = {"все": [], "пятерки": []}
    levels = [i / 10 for i in range(11)]
    sums = {k: [[0.0, 0.0, 0] for _ in levels] for k in curves}

    for key, tasks in benchmark_prior.PROFILES.items():
        truth = {m: snapshot.quality(key, m) for m in prices}
        present = [m for m, q in truth.items() if q is not None]
        if len(present) < MIN_MODELS:
            continue
        rng = random.Random(hash(key) % 100_000)
        train = without(snapshot, key)
        field = benchmark_prior.field_quality(train) or 0.5
        built = {m: element(m, prices[m], train, snapshot, field)[0] for m in present}
        pools = [("все", present)] + [("пятерки", rng.sample(present, POOL_SIZE)) for _ in range(POOLS_PER_SERIES)]
        for kind, pool in pools:
            for li, wq in enumerate(levels):
                weights = RouteWeights(max(wq, 0.05), max((1 - wq) / 2, 0.05), max((1 - wq) / 2, 0.05), 0.0)
                pick = score(variant, blind(tasks[0], os.environ.get("BLIND", "")), [built[m] for m in pool], weights)
                s = sums[kind][li]
                s[0] += truth[pick.name]
                s[1] += blended(prices[pick.name])
                s[2] += 1
    for kind, rows in sums.items():
        curves[kind] = [(levels[i], q / n, p / n) for i, (q, p, n) in enumerate(rows)]
    return curves


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    for rank, i in enumerate(order):
        result[i] = rank
    return result


def neighbours(train: BenchmarkSnapshot, model: str, vector, sharpness: float) -> float:
    """Качество модели в похожих сериях: среднее, взвешенное по сходству профилей задач."""
    import numpy as np

    weights, values = [], []
    for key, tasks in benchmark_prior.PROFILES.items():
        q = train.quality(key, model)
        if q is None:
            continue
        sim = float(np.dot(vector, tasks[0].feature_vector()))
        weights.append(math.exp(sharpness * sim))
        values.append(q)
    return sum(w * v for w, v in zip(weights, values)) / sum(weights) if weights else 0.0


def blind(task, mode: str):
    """Задача так, как ее видит хост без распознавания: известны только длины входа и ответа.
    «текст-нули» это нынешний GetFeatures (спецификация по умолчанию), «текст-типовая» подставляет
    спецификацию типовой задачи из профилей. Остальные режимы видят задачу целиком."""
    from fai_router.tracking import InputFeatures

    if mode == "текст-нули":
        return InputFeatures(input_len=task.input_len, len_answer=task.len_answer)
    if mode == "текст-типовая":
        typical = benchmark_prior.typical_task()
        return InputFeatures(input_len=task.input_len, len_answer=task.len_answer,
                             input_specifications=typical.input_specifications)
    return task


def accuracy(predictor: str = "приор") -> tuple[float, float, float]:
    """Точность прогноза на выброшенной серии: ранговая корреляция прогноза с истиной (средняя по
    сериям), доля серий, где прогнозный лидер и есть лучший, и средний недобор прогнозного лидера."""
    snapshot = default_snapshot()
    prices = popular()
    rhos, hits, regrets = [], [], []
    for key, tasks in benchmark_prior.PROFILES.items():
        truth = {m: snapshot.quality(key, m) for m in prices}
        present = [m for m, q in truth.items() if q is not None]
        if len(present) < MIN_MODELS:
            continue
        train = without(snapshot, key)
        field = benchmark_prior.field_quality(train) or 0.5
        vector = blind(tasks[0], predictor).feature_vector()
        if predictor in ("приор", "текст-нули", "текст-типовая"):
            predicted = [element(m, prices[m], train, snapshot, field)[0].get_quality_score(vector) for m in present]
        elif predictor == "общий":
            predicted = [statistics.fmean([q for _, q in benchmark_prior.measurements(train, m)] or [0.0]) for m in present]
        else:
            predicted = [neighbours(train, m, vector, float(predictor.split("-")[1])) for m in present]
        actual = [truth[m] for m in present]
        rp, ra = ranks(predicted), ranks(actual)
        mp, ma = statistics.fmean(rp), statistics.fmean(ra)
        cov = sum((x - mp) * (y - ma) for x, y in zip(rp, ra))
        rhos.append(cov / math.sqrt(sum((x - mp) ** 2 for x in rp) * sum((y - ma) ** 2 for y in ra)))
        leader = max(range(len(present)), key=lambda i: predicted[i])
        hits.append(1.0 if actual[leader] == max(actual) else 0.0)
        regrets.append(max(actual) - actual[leader])
    return statistics.fmean(rhos), statistics.fmean(hits), statistics.fmean(regrets)


def with_ridge(value: float) -> None:
    """Сила регуляризации приора: доля среднего квадрата длины задачи (QualityPrior)."""
    from fai_router.training import quality_prior
    fit = quality_prior.from_measurements
    benchmark_prior.from_measurements = lambda pairs, ridge=value: fit(pairs, ridge)


def with_bias(value: float) -> None:
    """Свободный член прогноза: постоянная координата после нормировки вектора задачи."""
    from fai_router.tracking import InputFeatures
    import numpy as np

    original = InputFeatures.feature_vector
    InputFeatures.feature_vector = lambda self: np.concatenate([original(self), [value]])


if __name__ == "__main__":
    import os
    if float(os.environ.get("BIAS", "0")) > 0:
        with_bias(float(os.environ["BIAS"]))
    if os.environ.get("RIDGE"):
        with_ridge(float(os.environ["RIDGE"]))
    variants = sys.argv[1:]
    if variants and variants[0] == "точность":
        for predictor in variants[1:] or ["приор"]:
            rho, hit, regret = accuracy(predictor)
            print(f"{predictor:10s} ранговая корреляция {rho:.3f}  лидер угадан {hit:.3f}  недобор лидера {regret:.3f}")
        sys.exit(0)
    if not variants:
        for name, (quality, price, regret, n) in sorted(run().items()):
            print(f"{name:32s} качество {quality:.3f}  недобор {regret:.3f}  цена ${price:6.2f}/М  n={n}")
    for variant in variants:
        for kind, curve in sweep(variant).items():
            print(f"{variant:14s} {kind:8s} " + "  ".join(f"{wq:.1f}:{q:.3f}/${p:.1f}" for wq, q, p in curve))
