"""Среда для соревнования кандидатов."""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, TypeVar

import numpy as np

from fai_router.enums import Capability
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import RouteWeights, Settings
from fai_router.tracking import InputFeatures, Tracert

T = TypeVar("T")

# Нижняя граница цены под логарифмом, доллары. Бесплатные модели дают нулевую стоимость,
# а логарифм нуля ушел бы в бесконечность и задавил бы разброс остальных кандидатов
COST_FLOOR = 1e-5


def route(text_prompt: str, elements: Iterable[RoutedElement], topk: int = 5,
          required: Capability = Capability.NONE, weights: RouteWeights | None = None) -> Tracert:
    """Полный ход роутинга: признаки запроса, соревнование кандидатов, трассировка.
    Баллы в трассировке проставляет судья, после того как победитель ответит.
    Веса этого выбора; пусто, тогда берутся общие из Settings."""
    candidates = list(elements)
    # Проверка до обращения к модели: распознавание задания стоит денег
    if not candidates:
        raise ValueError("Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям.")
    features = InputFeaturesService.get_features_full(text_prompt)
    return choose(features, candidates, topk, required, weights=weights)


def choose(features: InputFeatures, elements: Iterable[RoutedElement], topk: int = 5,
           required: Capability = Capability.NONE, rng: random.Random | None = None,
           weights: RouteWeights | None = None) -> Tracert:
    """Выбор исполнителя по готовым признакам, без обращения к модели. Из оценок топ-K строится
    распределение через softmax с температурой, и кандидат берется сэмплированием. Температура
    падает по мере накопления опыта: изученная группа выбирает почти жадно, группа с новичками
    пробует их чаще."""
    best = get_top_k(features, elements, topk, required, weights=weights)
    if not best:
        raise ValueError("Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям.")
    chosen = _sample(best, rng or random, weights)
    return Tracert(
        winner=best[chosen][1],
        top_k_elements=[element for _, element in best],
        input_feature_vector=features.feature_vector(),
        requested_spec=features.input_specifications,
        is_exploration=chosen != 0,
    )


def temperature(group: Iterable[RoutedElement], weights: RouteWeights | None = None) -> float:
    """Температура выбора: T = C/K * сумма корней из D*_k / m_k. Кандидат, о котором мало
    данных, поднимает температуру всей группы."""
    total, count = 0.0, 0
    for element in group:
        # Дисперсия по одному ходу равна нулю и означала бы уверенность на пустом месте,
        # поэтому кандидат считается изученным начиная со второго оцененного хода
        known = element.experience >= 2
        variance = element.score_variance if known else Settings.UNKNOWN_VARIANCE
        total += math.sqrt(variance / max(element.experience, 1))
        count += 1
    return 0.0 if count == 0 else (weights or Settings.current()).temperature_scale * total / count


def get_top_k(features: InputFeatures, elements: Iterable[RoutedElement], topk: int = 5,
              required: Capability = Capability.NONE,
              weights: RouteWeights | None = None) -> list[tuple[float, RoutedElement]]:
    """Лучшие по метрике R, отсев по возможностям идет до сравнения оценок.

    Метрика считается относительно группы. Качество, цена и время живут в несоизмеримых
    единицах, и в прежней формуле вклад цены был в 1156 раз меньше вклада качества: роутер
    выбирал, не глядя на деньги. Приведение каждого слагаемого к нулевому среднему и единичному
    разбросу внутри группы делает веса долями важности сравнимых величин."""
    w = weights or Settings.current()
    fit = [element for element in elements
           if element.supports(features.input_specifications, required)]
    if not fit:
        return []
    vector = features.feature_vector()
    quality = _standardize([element.get_quality_score(vector) for element in fit])
    cost = _standardize([math.log(element.get_cost(features) + COST_FLOOR) for element in fit])
    time = _standardize([math.log(element.get_time(features) + 2) for element in fit])

    # Слагаемые уже стандартизованы, поэтому масштаб суммы задают только веса: деление на
    # длину вектора весов делает R безразмерной величиной, а не зависящей от того, как именно
    # заданы WQ, WC и WT. Без этого температура выбора была откалибрована под один конкретный
    # набор весов и требовала перекалибровки при любом заметном их изменении.
    weight_norm = math.sqrt(w.WQ ** 2 + w.WC ** 2 + w.WT ** 2)
    normalizer = weight_norm if weight_norm > 1e-12 else 1.0

    scored = [
        ((w.WQ * quality[i] - w.WC * cost[i] - w.WT * time[i]) / normalizer, element)
        for i, element in enumerate(fit)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:topk]


def execute(trace: Tracert, run: Callable[[RoutedElement], T]) -> T:
    """Ход с запасными вариантами: если победитель отказал, работа переходит следующему из
    топ-K. Победитель в трассировке заменяется на того, кто справился, иначе похвалу получил бы
    кандидат, который ничего не сделал.

    Отмена (KeyboardInterrupt, asyncio.CancelledError) запасным вариантом не является: это
    BaseException, и мимо except Exception она проходит наверх, как и в версии на C#."""
    chain = [trace.winner] + [element for element in trace.top_k_elements if element is not trace.winner]
    failures: list[Exception] = []
    for candidate in chain:
        try:
            result = run(candidate)
        except Exception as error:  # noqa: BLE001 - отказ любого рода ведет к следующему кандидату
            failures.append(error)
            continue
        trace.winner = candidate
        return result
    raise RuntimeError(f"Отказали все {len(chain)} кандидатов из топ-K, ход выполнить некому: {failures}")


def _sample(best: list[tuple[float, RoutedElement]], rng, weights: RouteWeights | None) -> int:
    t = temperature((element for _, element in best), weights)
    # Температура ушла в ноль: группа изучена и разброса в отзывах нет, брать лучшего
    if t < 1e-9:
        return 0
    # Оценки сдвигаются на лучшую: при малой температуре показатель степени иначе улетает
    # в бесконечность, и распределение обращается в NaN
    top = best[0][0]
    chances = [math.exp((score - top) / t) for score, _ in best]
    dice = rng.random() * sum(chances)
    for index, chance in enumerate(chances):
        dice -= chance
        if dice <= 0:
            return index
    return 0


def _standardize(values: list[float]) -> np.ndarray:
    """Нулевое среднее и единичный разброс внутри группы. Одинаковые у всех значения дают
    нули: такое слагаемое на выбор не влияет, и это верно."""
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return np.zeros(len(array))
    deviation = array.std()
    return np.zeros(len(array)) if deviation < 1e-12 else (array - array.mean()) / deviation
