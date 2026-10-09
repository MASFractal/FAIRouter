"""Начальные веса кандидата по заранее замеренному качеству на задачах известных типов.

Без этого вектор кандидата задается по Ксавье, то есть случайно, и первый выбор роутера
оказывается жребием. Замер по типам задач делается один раз, зато система полезна с первого хода.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fai_router.settings import GLOBAL_MEAN, Settings

# Добавка к диагонали по умолчанию, доля среднего квадрата длины задачи (как DefaultRidge в C#)
DEFAULT_RIDGE = 0.3


def from_measurements(measurements: list[tuple[np.ndarray, float]], ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Вектор кандидата, при котором прогноз на замеренных задачах близок к замеру, в пространстве
    общего среднего Settings.task_mean.

    Замеров меньше, чем координат, поэтому решений бесконечно много и берется самое короткое.
    Оно лежит в линейной оболочке замеренных задач: v = сумма a_t x_t, а коэффициенты находятся
    из системы Грама.

    Добавка к диагонали это доля среднего квадрата длины задачи, а не абсолютное число: признаки
    бывают и нормированными, и сырыми. Прежняя добавка 1e-3 почти точно проводила прогноз через
    полсотни близких точек рейтингов, коэффициенты раздувались, и на реальных запросах прогноз
    уходил к 3 при шкале до 1. Доля 0,3 выбрана по ошибке «выбрось точку и предскажи ее» на
    рейтингах 263 моделей (21.09.2026): 0,241 при 1e-3, 0,182 при 0,3, 0,183 при 1."""
    return fit(measurements, None, GLOBAL_MEAN, ridge)


def fit(measurements: list[tuple[np.ndarray, float]], weights: Sequence[float] | None,
        mean: "np.ndarray | None | object", ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Как from_measurements, но с весами точек и явным средним задач (None значит без вычитания).

    Вес точки задает ее долю в ошибке подгонки: решается взвешенная гребневая задача, в двойственной
    форме (G + λ·diag(1/w))·α = q. Веса приводятся к среднему 1, поэтому сила стягивания та же, что
    без весов, а при равных весах решение совпадает с невзвешенным."""
    if not measurements:
        raise ValueError("Нужен хотя бы один замер.")
    if weights is not None and (len(weights) != len(measurements) or not all(weight > 0 for weight in weights)):
        raise ValueError("Весов должно быть столько же, сколько замеров, и все больше нуля.")
    # Тот же вид признаков, в котором работают прогноз и обучение
    tasks = np.array([Settings.center(task, mean) for task, _ in measurements])
    qualities = np.array([quality for _, quality in measurements], dtype=float)
    gram = tasks @ tasks.T
    scale = ridge * float(np.mean(np.diag(gram)))
    penalty = np.ones(len(tasks)) if weights is None else float(np.mean(weights)) / np.asarray(weights, dtype=float)
    gram = gram + scale * np.diag(penalty)
    try:
        coefficients = np.linalg.solve(gram, qualities)
    except np.linalg.LinAlgError:
        # Замеры повторяют друг друга, а добавки нет: берется решение наименьшей длины
        coefficients = np.linalg.lstsq(gram, qualities, rcond=None)[0]
    return coefficients @ tasks
