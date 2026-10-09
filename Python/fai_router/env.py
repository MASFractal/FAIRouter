"""Среда для соревнования кандидатов."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable, Iterable, TypeVar

import numpy as np

from fai_router.enums import Capability
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService, SpecSource
from fai_router.settings import RouteWeights, Settings, SufficiencyBar
from fai_router.tracking import InputFeatures, Tracert

T = TypeVar("T")

# Нижняя граница цены под логарифмом, доля самой низкой положительной цены группы. Бесплатные
# модели дают нулевую стоимость, а логарифм нуля ушел бы в бесконечность. Граница относительная:
# прежняя абсолютная 1e-5 доллара значила разное для рублевого каталога и для хода на тысячу
# токенов. Бесплатная модель выходит вдесятеро дешевле самой дешевой платной
COST_FLOOR_SHARE = 0.1

# Нижняя граница разброса логарифма цены и времени при стандартизации. Без нее группа, где цены
# различаются на проценты, раздувала эти проценты до единичного разброса, и копеечная разница
# решала ход наравне с десятикратной
MIN_LOG_DEVIATION = 0.25

# Нижняя граница разброса прогноза качества при стандартизации, по той же причине
MIN_QUALITY_DEVIATION = 0.05

# Вес качества среди прошедших планку, в долях от суммы весов цены и времени. Качество им уже
# обеспечено, и платить за лишнее незачем: решают цена и время, а качество остается только разнимать
# равных. Доля, а не число: постоянный вес 0,05 у заказчика, которому цена почти безразлична (вес
# цены тоже 0,05), уравнивал качество с ценой, и сильная модель снова выигрывала у достаточной. У
# профиля «только качество» (цена и время по нулю) доля дала бы ноль, и порядок задавал бы каталог;
# там качество получает вес единицу
SUFFICIENT_QUALITY_SHARE = 0.1

_EMPTY = "Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям."


@dataclass
class SufficientTop:
    """Итог выбора с планкой достаточности: лучшие кандидаты (прошедшие планку по метрике R либо,
    если не прошел никто, по вероятности достаточности, при равной по прогнозу качества, затем по
    имени) и дотянул ли кто-нибудь до планки."""

    top: list[tuple[float, RoutedElement]]
    reached: bool


def route(text_prompt: str, elements: Iterable[RoutedElement], topk: int = 5,
          required: Capability = Capability.NONE, weights: RouteWeights | None = None,
          turns: int = 1, bar: SufficiencyBar | None = None, specs: SpecSource | None = None,
          input_tokens: float | None = None, rng: random.Random | None = None) -> Tracert:
    """Полный ход роутинга: признаки запроса, соревнование кандидатов, трассировка.
    Баллы в трассировке проставляет судья, после того как победитель ответит.

    Веса этого выбора; пусто, тогда общие из Settings. С планкой bar выбор идет по принципу
    «необходимо и достаточно» (choose_sufficient), без нее по метрике R (choose). specs распознает
    задание (пусто, значит общий распознаватель через Settings.llm); input_tokens задает объем входа
    всего диалога. Сбой распознавания хода не роняет: выбор идет по типовой задаче, заказа в
    трассировке нет, и замер ответа пропускается, потому что сверять его не с чем."""
    candidates = list(elements)
    # Проверка до обращения к модели: распознавание задания стоит денег
    if not candidates:
        raise ValueError(_EMPTY)
    features = InputFeaturesService.get_features(text_prompt)
    recognized = InputFeaturesService.recognize(text_prompt, specs)
    if recognized is not None:
        InputFeaturesService.apply(features, recognized)
    features.turn_count = max(turns, 1)
    if input_tokens is not None and input_tokens > 0:
        features.input_len = input_tokens
    trace = (choose(features, candidates, topk, required, rng, weights) if bar is None
             else choose_sufficient(features, candidates, bar, topk, required, rng, weights))
    trace.requested_spec = recognized
    return trace


def choose(features: InputFeatures, elements: Iterable[RoutedElement], topk: int = 5,
           required: Capability = Capability.NONE, rng: random.Random | None = None,
           weights: RouteWeights | None = None) -> Tracert:
    """Выбор исполнителя по готовым признакам, без обращения к модели. Из оценок топ-K строится
    распределение через softmax с температурой, и кандидат берется сэмплированием. Температура
    падает по мере накопления опыта: изученная группа выбирает почти жадно, группа с новичками
    пробует их чаще."""
    best = get_top_k(features, elements, topk, required, weights=weights)
    if not best:
        raise ValueError(_EMPTY)
    return _trace(features, best, sample(best, weights, rng), required, None)


def choose_sufficient(features: InputFeatures, elements: Iterable[RoutedElement], bar: SufficiencyBar,
                      topk: int = 5, required: Capability = Capability.NONE,
                      rng: random.Random | None = None, weights: RouteWeights | None = None) -> Tracert:
    """Выбор с планкой достаточности, оформленный трассировкой: среди дотянувших до планки ход
    разыгрывается как в choose, по метрике R с температурой. Не дотянул никто, тогда жребий идет
    среди сильнейших по вероятности достаточности: без него при недоборе ход всегда доставался
    первому, остальные не набирали опыта, и выйти из недобора было не на чем."""
    chosen = get_sufficient(features, elements, bar, topk, required, weights=weights)
    if not chosen.top:
        raise ValueError(_EMPTY)
    return _trace(features, chosen.top, sample(chosen.top, weights, rng), required, chosen.reached)


def temperature(group: Iterable[RoutedElement], weights: RouteWeights | None = None) -> float:
    """Температура выбора: T = C/K * сумма корней из D*_k / m_k. Кандидат, о котором мало
    данных, поднимает температуру всей группы.

    m_k это условный опыт: отзывы людей, автоотзывы с весом и опыт, который стоят начальные веса по
    рейтингам (prior_experience). Без автоотзывов и рейтингов разведка не остывала, пока не придут
    человеческие отзывы, а их бывает ноль."""
    total, count = 0.0, 0
    for element in group:
        # Дисперсия по одному ходу равна нулю и означала бы уверенность на пустом месте,
        # поэтому кандидат считается изученным начиная со второго оцененного хода
        known = element.experience >= 2
        variance = element.score_variance if known else Settings.UNKNOWN_VARIANCE
        total += math.sqrt(variance / max(element.experience + element.prior_experience, 1))
        count += 1
    return 0.0 if count == 0 else (weights or Settings.current()).resolved_temperature_scale * total / count


def get_top_k(features: InputFeatures, elements: Iterable[RoutedElement], topk: int = 5,
              required: Capability = Capability.NONE,
              weights: RouteWeights | None = None) -> list[tuple[float, RoutedElement]]:
    """Лучшие по метрике R, отсев по возможностям и объему идет до сравнения оценок. Ничьи
    разбираются по прогнозу качества, затем по имени, поэтому порядок не зависит от каталога.

    Объем хода не вошел ни в одного кандидата с нужными возможностями: отдается один кандидат с
    наибольшим пределом, и трассировка это отмечает (context_shortfall). Прежде пул пустел, и ход
    падал исключением."""
    fit = _fit(features, elements, required)
    return _rank(features, fit, _quality(fit, features.feature_vector()), topk, weights or Settings.current())


def get_sufficient(features: InputFeatures, elements: Iterable[RoutedElement], bar: SufficiencyBar,
                   topk: int = 5, required: Capability = Capability.NONE,
                   weights: RouteWeights | None = None) -> SufficientTop:
    """Выбор по принципу «необходимо и достаточно»: отсеять тех, кто прогнозируемо не дотягивает
    до планки, а среди остальных взять дешевого и быстрого.

    Не прошел никто, тогда отдаются сильнейшие по вероятности достаточности, а reached ложно.
    Поднимать ли цену или предупредить человека, решает вызывающий: у него есть то, чего нет у
    библиотеки, то есть сам человек."""
    fit = _fit(features, elements, required)
    quality = _quality(fit, features.feature_vector())
    sufficiency = [bar.sufficiency(element.experience, quality[i]) for i, element in enumerate(fit)]
    passing = [i for i in range(len(fit)) if sufficiency[i] >= bar.bar]

    if passing:
        w = weights or Settings.current()
        share = SUFFICIENT_QUALITY_SHARE * (w.WC + w.WT)
        floored = replace(w, WQ=share if share > 1e-12 else 1.0)
        return SufficientTop(_rank(features, [fit[i] for i in passing], [quality[i] for i in passing],
                                   topk, floored), True)

    order = sorted(range(len(fit)), key=lambda i: (-sufficiency[i], -_finite_or_min(quality[i]), fit[i].name or ""))
    return SufficientTop([(sufficiency[i], fit[i]) for i in order[:topk]], False)


def execute(trace: Tracert, run: Callable[[RoutedElement], T]) -> T:
    """Ход с запасными вариантами: если победитель отказал, работа переходит следующему из
    топ-K. Победитель в трассировке заменяется на того, кто справился, иначе похвалу получил бы
    кандидат, который ничего не сделал; вместе с ним пересчитываются пометка разведки и прогноз.
    Пустой текст считается отказом. Отказавшие записываются в trace.failed.

    Отмена (KeyboardInterrupt, asyncio.CancelledError) запасным вариантом не является: это
    BaseException, и мимо except Exception она проходит наверх, как и в версии на C#."""
    chain = [trace.winner] + [element for element in trace.top_k_elements if element is not trace.winner]
    failures: list[Exception] = []
    for candidate in chain:
        try:
            result = run(candidate)
            if not _text_of(result).strip():
                raise ValueError(f"Кандидат {candidate.name} вернул пустой ответ.")
        except Exception as error:  # noqa: BLE001 - отказ любого рода ведет к следующему кандидату
            failures.append(error)
            trace.failed.append(candidate.name or "")
            continue
        if candidate is not trace.winner:
            trace.winner = candidate
            trace.is_exploration = bool(trace.top_k_elements) and candidate is not trace.top_k_elements[0]
            trace.forecast = candidate.get_quality_score(trace.input_feature_vector)
        return result
    raise RuntimeError(f"Отказали все {len(chain)} кандидатов из топ-K, ход выполнить некому: {failures}")


def sample(best: list[tuple[float, RoutedElement]], weights: RouteWeights | None = None,
           rng: random.Random | None = None) -> int:
    """Выбор кандидата сэмплированием из softmax по оценкам топ-K: индекс в списке, ноль при
    нулевой температуре. Открыт для вызывающих, которые строят топ-K сами."""
    if len(best) < 2:
        return 0
    t = temperature((element for _, element in best), weights)
    # Температура ушла в ноль: группа изучена и разброса в отзывах нет, брать лучшего
    if t < 1e-9:
        return 0
    # Оценки сдвигаются на наибольшую: при малой температуре показатель степени иначе улетает
    # в бесконечность, и распределение обращается в NaN
    top = max(score for score, _ in best)
    chances = [math.exp((score - top) / t) for score, _ in best]
    dice = (rng or random).random() * sum(chances)
    for index, chance in enumerate(chances):
        dice -= chance
        if dice <= 0:
            return index
    return 0


def _trace(features: InputFeatures, top: list[tuple[float, RoutedElement]], index: int,
           required: Capability, reached: bool | None) -> Tracert:
    vector = features.feature_vector()
    winner = top[index][1]
    return Tracert(
        winner=winner,
        top_k_elements=[element for _, element in top],
        input_feature_vector=vector,
        requested_spec=features.input_specifications,
        is_exploration=index != 0,
        forecast=winner.get_quality_score(vector),
        context_shortfall=not winner.supports(features, required),
        bar_reached=reached,
    )


def _fit(features: InputFeatures, elements: Iterable[RoutedElement], required: Capability) -> list[RoutedElement]:
    """Отсев по возможностям и объему до сравнения оценок: метрика R о возможностях ничего не знает,
    и кандидат, который заведомо не справится, выигрывал бы по цене и скорости."""
    able = [element for element in elements if element.can(features.input_specifications, required)]
    fit = [element for element in able if element.supports(features, required)]
    if fit or not able:
        return fit
    # Объем не вошел ни в кого: лучше ответ того, у кого предел больше, чем никакого
    return [max(able, key=lambda element: (_room(element.context_window), _room(element.context_limit)))]


def _room(limit: int) -> float:
    return limit if limit > 0 else math.inf


def _quality(fit: list[RoutedElement], vector: np.ndarray) -> list[float]:
    return [element.get_quality_score(vector) for element in fit]


def _rank(features: InputFeatures, fit: list[RoutedElement], quality: list[float], topk: int,
          w: RouteWeights) -> list[tuple[float, RoutedElement]]:
    """Метрика R относительно группы. Качество, цена и время живут в несоизмеримых единицах, и в
    прежней формуле вклад цены был в 1156 раз меньше вклада качества: роутер выбирал, не глядя на
    деньги. Приведение каждого слагаемого к нулевому среднему и единичному разбросу внутри группы
    делает веса долями важности сравнимых величин. Деление на длину вектора весов делает R
    безразмерной: температура не зависит от того, как именно заданы WQ, WC и WT."""
    costs = [element.get_cost(features) if element.has_known_price else math.nan for element in fit]
    floor = _cost_floor(costs)
    q = _standardize(quality, True, MIN_QUALITY_DEVIATION)
    cost = _standardize([math.log(value + floor) if math.isfinite(value) else math.nan for value in costs],
                        False, MIN_LOG_DEVIATION)
    time = _standardize([math.log(element.get_time(features) + 2) if element.tps > 0 else math.nan
                         for element in fit], False, MIN_LOG_DEVIATION)

    weight_norm = math.sqrt(w.WQ ** 2 + w.WC ** 2 + w.WT ** 2)
    normalizer = weight_norm if weight_norm > 1e-12 else 1.0
    scores = [(w.WQ * q[i] - w.WC * cost[i] - w.WT * time[i]) / normalizer for i in range(len(fit))]
    order = sorted(range(len(fit)), key=lambda i: (-scores[i], -_finite_or_min(quality[i]), fit[i].name or ""))
    return [(scores[i], fit[i]) for i in order[:topk]]


def _cost_floor(costs: list[float]) -> float:
    positive = [cost for cost in costs if math.isfinite(cost) and cost > 0]
    return COST_FLOOR_SHARE * min(positive) if positive else 1.0


def _standardize(values: list[float], low_is_worst: bool, min_deviation: float) -> np.ndarray:
    """Нулевое среднее и единичный разброс внутри группы. Нечисловое значение (неизвестная цена,
    NaN в векторе) ставится на шаг хуже худшего в группе, а не отравляет всю группу и не равняется
    с худшим известным. Разброс не меньше заданной границы: близкие значения не раздуваются до
    единичного разброса. Одинаковые у всех значения дают нули: на выбор такое слагаемое не влияет."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(array) < 2 or len(finite) == 0:
        return np.zeros(len(array))
    step = max(float(finite.max() - finite.min()), min_deviation)
    worst = finite.min() - step if low_is_worst else finite.max() + step
    clean = np.where(np.isfinite(array), array, worst)
    deviation = max(float(clean.std()), min_deviation)
    return (clean - clean.mean()) / deviation


def _finite_or_min(value: float) -> float:
    return value if math.isfinite(value) else -math.inf


def _text_of(result: object) -> str:
    """Текст ответа исполнителя: строка или объект с полем text (Completion); иной результат
    пустым не считается, кроме None."""
    if result is None or isinstance(result, str):
        return result or ""
    return str(getattr(result, "text", "результат") or "")
