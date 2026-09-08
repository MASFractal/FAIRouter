from __future__ import annotations

import numpy as np

from fai_router.diff_spec import DiffSpec
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import Tracert


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


class Judge:
    """Судья, автоматическая оценка качества решения. Замером факта судья не занимается:
    обе спецификации приходят к нему готовыми."""

    def __init__(self):
        # Матрица трансформации (обучаемая). Единичная на старте означает наивное допущение,
        # будто факт обязан совпасть с заказом
        self.transformer_w: np.ndarray = np.eye(Settings.features_spec_dim())

    def get_score(self, input_spec: Specifications, actual_spec: Specifications) -> float:
        """Близость факта к заказу, пропущенному через обучаемую матрицу."""
        transformed = self.transformer_w @ input_spec.feature_vector()
        return cosine(actual_spec.feature_vector(), transformed)

    def rate(self, trace: Tracert, input_spec: Specifications, actual_spec: Specifications) -> float:
        """Проставляет оценку хода в его трассировку и тем замыкает круг: ход состоялся,
        судья его оценил, и трассировка стала обучающим примером."""
        trace.score = self.get_score(input_spec, actual_spec)
        return trace.score

    @staticmethod
    def criticize(input_spec: Specifications, actual_spec: Specifications) -> DiffSpec:
        """Режим критика: расхождения между заданием и фактом по каждому пункту."""
        return DiffSpec.compare(input_spec, actual_spec)
