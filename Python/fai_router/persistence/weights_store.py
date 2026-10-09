from __future__ import annotations

import json
import logging
import sqlite3
from typing import Iterable

import numpy as np

from fai_router.judge import Judge
from fai_router.persistence.db import connect, ensure_schema
from fai_router.routed_element import RoutedElement
from fai_router.settings import Settings

log = logging.getLogger("fai_router")


class SqliteWeightsStore:
    """Хранилище обученных весов: векторы кандидатов вместе с их средним задач и матрица судьи.
    Кандидат опознается по имени, поэтому переименование равносильно потере обучения.

    Сохранять стоит только векторы, которых касалось обучение: вектор, записанный необученным, при
    загрузке затирал бы свежий начальный вектор по новому снимку рейтингов. Это решает вызывающий
    (FaiRouter.save), хранилище пишет то, что дали."""

    def __init__(self, database_path: str):
        self._path = database_path
        ensure_schema(database_path)

    def save(self, elements: Iterable[RoutedElement], judge: Judge | None = None) -> None:
        """Сохраняет векторы кандидатов и матрицу судьи одной транзакцией: база не остается с новыми
        векторами и старой матрицей, если запись прервалась посередине."""
        with connect(self._path) as connection:
            for element in elements:
                _save_vector(connection, element)
            if judge is not None:
                _save_judge(connection, judge)

    def save_elements(self, elements: Iterable[RoutedElement]) -> None:
        """Сохраняет только векторы кандидатов, одной транзакцией."""
        self.save(elements)

    def load_vectors(self, elements: Iterable[RoutedElement]) -> list[str]:
        """Восстанавливает векторы и их среднее задач. Возвращает имена узнанных кандидатов;
        незнакомые остаются с начальными весами.

        Вектор другой размерности пропускается с предупреждением, а не роняет загрузку всех: признаки
        сменили длину, и обучение этого кандидата к ним не относится, а остальные от этого не портятся.
        Вектор, сохраненный до того, как среднее стало храниться при нем, получает общее среднее из
        таблицы task_mean: именно в нем его тогда учили."""
        restored = []
        with connect(self._path) as connection:
            legacy_mean = _read_task_mean(connection)
            for element in elements:
                row = connection.execute("SELECT values_json, mean_json FROM element_vectors WHERE name = ?",
                                         (element.name or "",)).fetchone()
                if row is None:
                    continue
                values = np.array(json.loads(row[0]), dtype=float)
                mean = legacy_mean if row[1] is None else np.array(json.loads(row[1]), dtype=float)
                if (len(values) != len(element.ideal_match_vector) or not np.isfinite(values).all()
                        or (mean is not None and len(mean) != len(values))):
                    log.warning("Вектор кандидата «%s» в базе размерности %d, а ожидается %d, либо он поврежден; "
                                "кандидат остается с начальными весами.", element.name, len(values),
                                len(element.ideal_match_vector))
                    continue
                element.ideal_match_vector = values
                element.task_mean = mean
                restored.append(element.name)
        return restored

    def load_elements(self, elements: Iterable[RoutedElement]) -> int:
        """Возвращает число узнанных кандидатов, незнакомые и чужой размерности остаются с начальными весами."""
        return len(self.load_vectors(elements))

    def save_judge(self, judge: Judge) -> None:
        with connect(self._path) as connection:
            _save_judge(connection, judge)

    def load_judge(self, judge: Judge) -> bool:
        """Ложь означает, что обученной матрицы в базе нет или она другой размерности (тогда с
        предупреждением остается нынешняя)."""
        with connect(self._path) as connection:
            row = connection.execute(
                "SELECT rows_count, columns_count, values_json FROM judge_matrix WHERE id = 1").fetchone()
        if row is None:
            return False
        rows, columns, values = row[0], row[1], np.array(json.loads(row[2]), dtype=float)
        if (rows, columns) != judge.transformer_w.shape or len(values) != rows * columns:
            log.warning("Матрица судьи в базе %dx%d, а ожидается %dx%d; признаки изменились, судья остается "
                        "с нынешней матрицей.", rows, columns, *judge.transformer_w.shape)
            return False
        judge.transformer_w = values.reshape(rows, columns)
        return True

    def save_task_mean(self, task_mean: np.ndarray) -> None:
        """Сохраняет общее среднее по векторам задач. Оставлено ради совместимости: среднее теперь
        хранится при каждом векторе."""
        with connect(self._path) as connection:
            connection.execute(
                "INSERT INTO task_mean (id, values_json) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET values_json = excluded.values_json",
                (json.dumps(np.asarray(task_mean).tolist()),),
            )

    def load_task_mean(self) -> np.ndarray | None:
        """Общее среднее по векторам задач. None, если его не сохраняли или оно другой размерности,
        чем нынешние признаки."""
        with connect(self._path) as connection:
            return _read_task_mean(connection)


def _read_task_mean(connection: sqlite3.Connection) -> np.ndarray | None:
    row = connection.execute("SELECT values_json FROM task_mean WHERE id = 1").fetchone()
    if row is None:
        return None
    values = np.array(json.loads(row[0]), dtype=float)
    if len(values) == Settings.full_dim():
        return values
    log.warning("Среднее задач в базе размерности %d, а ожидается %d; оно не применяется.",
                len(values), Settings.full_dim())
    return None


def _save_vector(connection: sqlite3.Connection, element: RoutedElement) -> None:
    if not element.name or not element.name.strip():
        raise ValueError("Кандидат без имени: вес не под чем сохранять.")
    mean = Settings.task_mean if element.task_mean is None else element.task_mean
    connection.execute(
        "INSERT INTO element_vectors (name, dimension, values_json, mean_json) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET dimension = excluded.dimension, values_json = excluded.values_json, "
        "mean_json = excluded.mean_json",
        (element.name, len(element.ideal_match_vector), json.dumps(element.ideal_match_vector.tolist()),
         None if mean is None else json.dumps(np.asarray(mean).tolist())),
    )


def _save_judge(connection: sqlite3.Connection, judge: Judge) -> None:
    matrix = judge.transformer_w
    connection.execute(
        "INSERT INTO judge_matrix (id, rows_count, columns_count, values_json) VALUES (1, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET rows_count = excluded.rows_count, "
        "columns_count = excluded.columns_count, values_json = excluded.values_json",
        (matrix.shape[0], matrix.shape[1], json.dumps(matrix.ravel().tolist())),
    )
