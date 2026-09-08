from __future__ import annotations

import numpy as np

from fai_router.judge import Judge
from fai_router.specifications import Specifications


class JudgeTrainer:
    """Обучение судьи: матрица трансформации подгоняется так, чтобы оценка судьи повторяла
    оценку человека. Ошибкой служит квадрат расхождения, обучение идет спуском по градиенту.

    В версии на C# градиент считал автоград из AI.NeuralNetworks. Здесь он выведен аналитически:
    для косинуса между фактом f и преобразованным заказом u = W r производная по W равна
    внешнему произведению производной по u на r."""

    def __init__(self, judge: Judge, learning_rate: float = 0.01):
        self._judge = judge
        self._lr = learning_rate

    def train(self, requested: Specifications, actual: Specifications, human_score: float) -> float:
        """Один шаг обучения. Возвращает ошибку до шага: по ее убыванию видно, что судья учится."""
        eps = 1e-8
        r = requested.feature_vector()
        f = actual.feature_vector()
        w = self._judge.transformer_w

        u = w @ r
        norm_f = np.sqrt(f @ f + eps * eps)
        norm_u = np.sqrt(u @ u + eps * eps)
        score = (f @ u) / (norm_f * norm_u)
        loss = (score - human_score) ** 2

        d_score = 2 * (score - human_score)
        d_u = f / (norm_f * norm_u) - (f @ u) * u / (norm_f * norm_u ** 3)
        gradient = np.outer(d_score * d_u, r)

        self._judge.transformer_w = w - self._lr * gradient
        return float(loss)
