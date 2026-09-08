from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fai_router.enums import FeedbackType
from fai_router.settings import Settings
from fai_router.specifications import Specifications

if TYPE_CHECKING:
    from fai_router.routed_element import RoutedElement


class InputFeatures:
    """Признаки входа: объем запроса и ожидаемого ответа плюс распознанное задание."""

    # Типичные объемы в токенах. Счетчики входят в вектор через логарифмическую шкалу, как и
    # объемы в спецификации. В сырых токенах они давали 100% длины вектора на настоящем
    # запросе, и все координаты спецификации весили ноль: роутер не видел типа задачи
    INPUT_SCALE = 2000.0
    ANSWER_SCALE = 7000.0

    def __init__(self, input_len: float = 0.0, len_answer: float = 0.0,
                 input_specifications: Specifications | None = None):
        self.input_len = input_len
        self.len_answer = len_answer
        self.input_specifications = Specifications() if input_specifications is None else input_specifications

    def feature_vector(self) -> np.ndarray:
        """Объемные признаки и вектор спецификации, приведенные к единичной длине."""
        head = np.array([
            Specifications.scaled(self.input_len, self.INPUT_SCALE),
            Specifications.scaled(self.len_answer, self.ANSWER_SCALE),
        ])
        vector = np.concatenate([head, self.input_specifications.feature_vector()])
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-12 else vector / norm


@dataclass
class Tracert:
    """Трассировка хода: обучающий пример для роутера."""

    winner: "RoutedElement"
    top_k_elements: list["RoutedElement"]
    input_feature_vector: np.ndarray
    # Заказанная спецификация, распознанная при построении признаков. Судье она нужна
    # для оценки хода, без нее пришлось бы обращаться к модели повторно
    requested_spec: Specifications | None = None
    # Ход отдан не лидеру, а сопернику. Без пометки при разборе накопленного нельзя
    # отличить осознанный выбор роутера от жребия
    is_exploration: bool = False
    # Баллы за задачу. Проставляет судья после того, как победитель ответил
    score: float = 0.0


@dataclass
class Feedback:
    """Отзыв на результат."""

    ftype: FeedbackType = FeedbackType.AUTO
    score: float = 0.0
