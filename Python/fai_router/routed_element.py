from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

import numpy as np

from fai_router.enums import Capability
from fai_router.settings import Settings
from fai_router.specifications import Specifications

if TYPE_CHECKING:
    from fai_router.tracking import InputFeatures


class RoutedElement:
    """Кандидат на исполнение: цены, скорость, возможности и обучаемый вектор соответствия.

    Числа проверяются при записи: NaN и бесконечность отвергаются сразу, потому что в выборе они
    отравляли бы стандартизацию всей группы. Отрицательная цена допустима и означает, что поставщик
    ее не зафиксировал (has_known_price): такой кандидат в выборе получает худшую цену группы."""

    def __init__(
        self,
        name: str,
        tps: float = 1.0,
        dpmt_inp: float = 0.0,
        dpmt_outp: float = 0.0,
        capabilities: Capability = Capability.ALL,
        context_limit: int = 0,
        ideal_match_vector: np.ndarray | None = None,
        context_window: int = 0,
        task_mean: np.ndarray | None = None,
    ):
        self.name = name
        # Число токенов в секунду; ноль и меньше означает, что скорость неизвестна
        self.tps = tps
        # Цена за миллион токенов, вход и выход; отрицательная означает, что цена неизвестна
        self.dpmt_inp = dpmt_inp
        self.dpmt_outp = dpmt_outp
        # Что кандидат умеет. По умолчанию все: ограничения задает вызывающий
        self.capabilities = capabilities
        # Наибольший объем ОТВЕТА в символах, ноль означает без ограничения
        self.context_limit = context_limit
        # Окно контекста в токенах: вход всего диалога вместе с ожидаемым ответом должен в него
        # поместиться. Ноль означает, что окно неизвестно, и проверки нет
        self.context_window = context_window
        # Вектор для сравнения (обучаемый), инициализация по Ксавье. Обучение подменяет его
        # целиком, поэтому соседний выбор видит старый вектор или новый, но не смесь
        self.ideal_match_vector = (
            self.xavier_vector(Settings.full_dim()) if ideal_match_vector is None else ideal_match_vector
        )
        # Среднее задач, в пространстве которого построен вектор; None значит общее Settings.task_mean
        self.task_mean = task_mean
        # Условный опыт: человеческие отзывы целиком, автоотзывы с весом AUTO_FEEDBACK_WEIGHT; m_k
        # в формуле температуры
        self.experience = 0.0
        # Опыт, который стоят начальные веса: у кандидата с весами по рейтингам он равен
        # Settings.PRIOR_EXPERIENCE. Входит в знаменатель температуры, но не в оценку дисперсии
        self.prior_experience = 0.0
        # Оценка дисперсии отзывов, в формуле температуры это D*_k
        self.score_variance = 0.0
        # Во сколько раз ход обходится дороже прайса: переделки после отказа приемки. Единица
        # означает «как по прайсу»; поправку задает вызывающий по накопленным ходам
        self.cost_ratio = 1.0

    @property
    def ideal_match_vector(self) -> np.ndarray:
        return self._vector

    @ideal_match_vector.setter
    def ideal_match_vector(self, value: np.ndarray) -> None:
        vector = np.asarray(value, dtype=float)
        if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all():
            raise ValueError("Вектор кандидата пуст или содержит нечисловые значения.")
        self._vector = vector

    @property
    def tps(self) -> float:
        return self._tps

    @tps.setter
    def tps(self, value: float) -> None:
        self._tps = _finite(value, "tps")

    @property
    def dpmt_inp(self) -> float:
        return self._dpmt_inp

    @dpmt_inp.setter
    def dpmt_inp(self, value: float) -> None:
        self._dpmt_inp = _finite(value, "dpmt_inp")

    @property
    def dpmt_outp(self) -> float:
        return self._dpmt_outp

    @dpmt_outp.setter
    def dpmt_outp(self, value: float) -> None:
        self._dpmt_outp = _finite(value, "dpmt_outp")

    @property
    def cost_ratio(self) -> float:
        return self._cost_ratio

    @cost_ratio.setter
    def cost_ratio(self, value: float) -> None:
        if _finite(value, "cost_ratio") <= 0:
            raise ValueError(f"Поправка цены должна быть больше нуля, задано {value}.")
        self._cost_ratio = float(value)

    @property
    def has_known_price(self) -> bool:
        """Цена известна: обе ставки не отрицательны."""
        return self.dpmt_inp >= 0 and self.dpmt_outp >= 0

    def get_quality_score(self, features: np.ndarray) -> float:
        """Прогноз качества: скалярное произведение признаков задачи на вектор кандидата в
        пространстве его среднего задач."""
        mean = Settings.task_mean if self.task_mean is None else self.task_mean
        return float(Settings.center(features, mean) @ self.ideal_match_vector)

    def supports(self, target: "Specifications | InputFeatures | None",
                 required: Capability = Capability.NONE) -> bool:
        """Справится ли кандидат с заданием в принципе: возможности и объем ответа, а по признакам
        хода еще и окно контекста для всего диалога вместе с ответом. Проверка намеренно грубая:
        она отсеивает заведомо непригодных, а не выбирает лучшего."""
        specifications = target if target is None or isinstance(target, Specifications) else target.input_specifications
        if not self.can(specifications, required):
            return False
        if specifications is not None and 0 < self.context_limit < specifications.symbol_length:
            return False
        if target is specifications or self.context_window <= 0:
            return True
        return target.input_len + target.len_answer <= self.context_window

    def can(self, specifications: Specifications | None, required: Capability = Capability.NONE) -> bool:
        """Есть ли у кандидата нужные возможности, без проверки объема."""
        if specifications is not None:
            if specifications.code_block_count > 0:
                required |= Capability.CODE
            if specifications.formula_count > 0:
                required |= Capability.FORMULAS
        return (self.capabilities & required) == required

    def get_cost(self, features: InputFeatures) -> float:
        """Ожидаемая стоимость запроса: прайс с поправкой на то, во что ход обходится на деле."""
        return self.get_list_cost(features) * self.cost_ratio

    def get_list_cost(self, features: InputFeatures) -> float:
        """Стоимость по прайсу, без поправки. С ней сравнивается фактическая цена хода, поэтому
        поправка сюда не входит: иначе она считалась бы сама из себя."""
        return (self.dpmt_inp * features.input_len + self.dpmt_outp * features.len_answer) * 1e-6

    def get_time(self, features: InputFeatures) -> float:
        """Ожидаемое время ответа в секундах."""
        return features.len_answer / (self.tps + 0.1)

    @staticmethod
    def xavier_vector(dimension: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Равномерно из [-limit, limit], limit = sqrt(6 / n). Разброс задан размерностью,
        поэтому прогноз на старте не зависит от числа признаков."""
        rng = np.random.default_rng() if rng is None else rng
        limit = math.sqrt(6.0 / dimension)
        return rng.uniform(-limit, limit, dimension)

    def __repr__(self) -> str:
        return f"RoutedElement({self.name!r})"


def by_name(elements: Iterable[RoutedElement]) -> dict[str, RoutedElement]:
    """Кандидаты по имени. Безымянные пропускаются, из повторов имени берется первый: повтор модели
    в списке раньше ронял журнал и обучение."""
    named: dict[str, RoutedElement] = {}
    for element in elements:
        if element.name and element.name.strip():
            named.setdefault(element.name, element)
    return named


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}: нужно конечное число, задано {value}.")
    return number
