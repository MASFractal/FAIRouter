from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Calibration:
    """Калибровка прогноза качества в вероятность того, что ответ устроит человека: p = σ(A·q + B).

    Прогноз кандидата относительный: скалярное произведение признаков задачи на его вектор умеет
    сказать «этот лучше того», но не «этот сойдет». Планке достаточности нужна абсолютная шкала, и
    калибровка переводит прогноз в долю лайков, которую такой прогноз получал на деле.

    Решатель тот же, что в версии на C#, шаг в шаг: метод Ньютона на системе 2×2, поэтому на одних
    и тех же парах обе версии дают одни и те же A и B."""

    A: float
    B: float

    # Сколько шагов Ньютона разрешено: на двух параметрах сходится за десяток
    MAX_ITERATIONS = 50
    # Шаг, меньше которого решение считается найденным
    TOLERANCE = 1e-10
    # Слабая привязка сдвига к доле лайков. Когда все оценки одинаковые, у сдвига нет конечного
    # оптимума, и без привязки он уходил бы в бесконечность
    SHIFT_RIDGE = 1e-3

    def predict(self, quality: float) -> float:
        """Вероятность, что ответ с таким прогнозом устроит человека."""
        return _sigmoid(self.A * quality + self.B)

    @classmethod
    def fit(cls, pairs: Iterable[tuple[float, float]], ridge: float = 1.0) -> "Calibration":
        """Подбирает калибровку по парам «прогноз и оценка». Наклон стягивается к нулю: пока
        оценок мало, прогноз считается малоинформативным, и калибровка отдает почти одну долю
        лайков, а не выдумывает зависимость."""
        pairs = list(pairs)
        if not pairs:
            raise ValueError("Нужна хотя бы одна оценка.")

        rate = min(max(sum(score for _, score in pairs) / len(pairs), 1e-3), 1 - 1e-3)
        anchor = math.log(rate / (1 - rate))
        a, b = 0.0, anchor

        for _ in range(cls.MAX_ITERATIONS):
            g_a, g_b = ridge * a, cls.SHIFT_RIDGE * (b - anchor)
            h_aa, h_ab, h_bb = ridge, 0.0, cls.SHIFT_RIDGE

            for q, y in pairs:
                p = _sigmoid(a * q + b)
                w = p * (1 - p)
                g_a += (p - y) * q
                g_b += p - y
                h_aa += w * q * q
                h_ab += w * q
                h_bb += w

            det = h_aa * h_bb - h_ab * h_ab
            if det < 1e-18:
                break

            d_a = (h_bb * g_a - h_ab * g_b) / det
            d_b = (h_aa * g_b - h_ab * g_a) / det
            a -= d_a
            b -= d_b

            if abs(d_a) < cls.TOLERANCE and abs(d_b) < cls.TOLERANCE:
                break

        return cls(a, b)


def _sigmoid(x: float) -> float:
    # Показатель ограничен: на краях экспонента иначе дает бесконечность
    return 1.0 / (1.0 + math.exp(-min(max(x, -35.0), 35.0)))
