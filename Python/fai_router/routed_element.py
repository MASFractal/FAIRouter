from __future__ import annotations

import math

import numpy as np

from fai_router.enums import Capability
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import InputFeatures


class RoutedElement:
    """Кандидат на исполнение: цены, скорость, возможности и обучаемый вектор соответствия."""

    def __init__(
        self,
        name: str,
        tps: float = 1.0,
        dpmt_inp: float = 0.0,
        dpmt_outp: float = 0.0,
        capabilities: Capability = Capability.ALL,
        context_limit: int = 0,
        ideal_match_vector: np.ndarray | None = None,
    ):
        self.name = name
        # Число токенов в секунду
        self.tps = tps
        # Цена в долларах за миллион токенов, вход и выход
        self.dpmt_inp = dpmt_inp
        self.dpmt_outp = dpmt_outp
        # Что кандидат умеет. По умолчанию все: ограничения задает вызывающий
        self.capabilities = capabilities
        # Наибольший объем ответа в символах, ноль означает без ограничения
        self.context_limit = context_limit
        # Вектор для сравнения (обучаемый), инициализация по Ксавье
        self.ideal_match_vector = (
            self.xavier_vector(Settings.full_dim()) if ideal_match_vector is None else ideal_match_vector
        )
        # Число оцененных ходов, в формуле температуры это m_k
        self.experience = 0
        # Оценка дисперсии отзывов, в формуле температуры это D*_k
        self.score_variance = 0.0

    def get_quality_score(self, features: np.ndarray) -> float:
        """Прогноз качества: скалярное произведение признаков задачи на вектор кандидата."""
        return float(Settings.center(features) @ self.ideal_match_vector)

    def supports(self, specifications: Specifications | None, required: Capability = Capability.NONE) -> bool:
        """Справится ли кандидат с таким заданием в принципе. Проверка намеренно грубая:
        она отсеивает заведомо непригодных, а не выбирает лучшего."""
        if specifications is not None:
            if specifications.code_block_count > 0:
                required |= Capability.CODE
            if specifications.formula_count > 0:
                required |= Capability.FORMULAS
            if self.context_limit > 0 and specifications.symbol_length > self.context_limit:
                return False
        return (self.capabilities & required) == required

    def get_cost(self, features: InputFeatures) -> float:
        """Стоимость запроса в долларах по ценам кандидата."""
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
