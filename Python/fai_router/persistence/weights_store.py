from __future__ import annotations

import json
import sqlite3
from typing import Iterable

import numpy as np

from fai_router.judge import Judge
from fai_router.routed_element import RoutedElement


class SqliteWeightsStore:
    """Хранилище обученных весов: векторы кандидатов, матрица судьи и среднее по задачам.
    Кандидат опознается по имени, поэтому переименование равносильно потере обучения."""

    def __init__(self, database_path: str):
        self._path = database_path
        with self._open() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS element_vectors (
                    name TEXT PRIMARY KEY, dimension INTEGER NOT NULL, values_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS task_mean (
                    id INTEGER PRIMARY KEY CHECK (id = 1), values_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS judge_matrix (
                    id INTEGER PRIMARY KEY CHECK (id = 1), rows_count INTEGER NOT NULL,
                    columns_count INTEGER NOT NULL, values_json TEXT NOT NULL);
            """)

    def save_elements(self, elements: Iterable[RoutedElement]) -> None:
        with self._open() as connection:
            for element in elements:
                if not element.name or not element.name.strip():
                    raise ValueError("Кандидат без имени: вес не под чем сохранять.")
                connection.execute(
                    "INSERT INTO element_vectors (name, dimension, values_json) VALUES (?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET dimension = excluded.dimension, values_json = excluded.values_json",
                    (element.name, len(element.ideal_match_vector), json.dumps(element.ideal_match_vector.tolist())),
                )

    def load_elements(self, elements: Iterable[RoutedElement]) -> int:
        """Возвращает число узнанных кандидатов, незнакомые остаются с начальными весами."""
        restored = 0
        with self._open() as connection:
            for element in elements:
                row = connection.execute(
                    "SELECT values_json FROM element_vectors WHERE name = ?", (element.name,)).fetchone()
                if row is None:
                    continue
                values = np.array(json.loads(row[0]))
                _expect(len(values), len(element.ideal_match_vector), f"вектор кандидата «{element.name}»")
                element.ideal_match_vector = values
                restored += 1
        return restored

    def save_judge(self, judge: Judge) -> None:
        matrix = judge.transformer_w
        with self._open() as connection:
            connection.execute(
                "INSERT INTO judge_matrix (id, rows_count, columns_count, values_json) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET rows_count = excluded.rows_count, "
                "columns_count = excluded.columns_count, values_json = excluded.values_json",
                (matrix.shape[0], matrix.shape[1], json.dumps(matrix.ravel().tolist())),
            )

    def load_judge(self, judge: Judge) -> bool:
        """Ложь означает, что обученной матрицы в базе еще нет."""
        with self._open() as connection:
            row = connection.execute(
                "SELECT rows_count, columns_count, values_json FROM judge_matrix WHERE id = 1").fetchone()
        if row is None:
            return False
        rows, columns, values = row[0], row[1], np.array(json.loads(row[2]))
        _expect(rows, judge.transformer_w.shape[0], "число строк матрицы судьи")
        _expect(columns, judge.transformer_w.shape[1], "число столбцов матрицы судьи")
        judge.transformer_w = values.reshape(rows, columns)
        return True

    def save_task_mean(self, task_mean: np.ndarray) -> None:
        with self._open() as connection:
            connection.execute(
                "INSERT INTO task_mean (id, values_json) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET values_json = excluded.values_json",
                (json.dumps(task_mean.tolist()),),
            )

    def load_task_mean(self) -> np.ndarray | None:
        """Загружать среднее нужно вместе с весами и до первого хода: веса обучены в
        пространстве с этим средним."""
        with self._open() as connection:
            row = connection.execute("SELECT values_json FROM task_mean WHERE id = 1").fetchone()
        return None if row is None else np.array(json.loads(row[0]))

    def _open(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)


def _expect(stored: int, expected: int, what: str) -> None:
    # Размерность признаков меняется при добавлении стиля или метрики: молча взятые веса
    # прежней длины означали бы обучение поверх мусора
    if stored != expected:
        raise ValueError(
            f"В базе {what} размерности {stored}, а сейчас ожидается {expected}. "
            "Признаки изменились, сохраненные веса больше не применимы.")
