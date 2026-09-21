"""Начальные веса кандидата по заранее замеренному качеству на задачах известных типов.

Без этого вектор кандидата задается по Ксавье, то есть случайно, и первый выбор роутера
оказывается жребием. Замер по типам задач делается один раз, зато система полезна с первого хода.
"""

from __future__ import annotations

import numpy as np

from fai_router.settings import Settings

# Добавка к диагонали по умолчанию, доля среднего квадрата длины задачи (как DefaultRidge в C#)
DEFAULT_RIDGE = 0.3


def from_measurements(measurements: list[tuple[np.ndarray, float]], ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Вектор кандидата, при котором прогноз на замеренных задачах близок к замеру.

    Замеров меньше, чем координат, поэтому решений бесконечно много и берется самое короткое.
    Оно лежит в линейной оболочке замеренных задач: v = сумма a_t x_t, а коэффициенты находятся
    из системы Грама.

    Добавка к диагонали — доля среднего квадрата длины задачи, а не абсолютное число: признаки
    бывают и нормированными, и сырыми. Прежняя добавка 1e-3 почти точно проводила прогноз через
    полсотни близких точек рейтингов, коэффициенты раздувались, и на реальных запросах прогноз
    уходил к 3 при шкале до 1. Доля 0,3 выбрана по ошибке «выбрось точку и предскажи ее» на
    рейтингах 263 моделей (21.09.2026): 0,241 при 1e-3, 0,182 при 0,3, 0,183 при 1."""
    if not measurements:
        raise ValueError("Нужен хотя бы один замер.")
    tasks = np.array([Settings.center(task) for task, _ in measurements])
    qualities = np.array([quality for _, quality in measurements], dtype=float)
    gram = tasks @ tasks.T
    gram = gram + ridge * float(np.mean(np.diag(gram))) * np.eye(len(tasks))
    coefficients = np.linalg.solve(gram, qualities)
    return coefficients @ tasks
