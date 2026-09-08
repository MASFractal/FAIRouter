"""Начальные веса кандидата по заранее замеренному качеству на задачах известных типов.

Без этого вектор кандидата задается по Ксавье, то есть случайно, и первый выбор роутера
оказывается жребием. Замер по типам задач делается один раз, зато система полезна с первого хода.
"""

from __future__ import annotations

import numpy as np

from fai_router.settings import Settings


def from_measurements(measurements: list[tuple[np.ndarray, float]], ridge: float = 1e-3) -> np.ndarray:
    """Вектор кандидата, при котором прогноз на замеренных задачах совпадает с замером.

    Замеров меньше, чем координат, поэтому решений бесконечно много и берется самое короткое.
    Оно лежит в линейной оболочке замеренных задач: v = сумма a_t x_t, а коэффициенты находятся
    из системы Грама. Добавка к диагонали удерживает решение, когда задачи почти сонаправлены."""
    if not measurements:
        raise ValueError("Нужен хотя бы один замер.")
    tasks = np.array([Settings.center(task) for task, _ in measurements])
    qualities = np.array([quality for _, quality in measurements], dtype=float)
    gram = tasks @ tasks.T + ridge * np.eye(len(tasks))
    coefficients = np.linalg.solve(gram, qualities)
    return coefficients @ tasks
