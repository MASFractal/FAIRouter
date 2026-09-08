from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from fai_router.enums import FeedbackType
from fai_router.routed_element import RoutedElement
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert


@dataclass
class TrainingRound:
    """Ход, накопленный для обучения: трассировка, отзыв и пара «заказ и факт»."""

    id: int
    trace: Tracert
    feedback: Feedback
    requested: Specifications | None
    actual: Specifications | None


class SqliteTraceStore:
    """Накопитель ходов и отзывов в той же базе, что и веса. Тренеры делают шаг по одному
    примеру, а закономерность видна только на выборке. Отзыв приходит позже хода, поэтому
    пишется отдельным действием."""

    def __init__(self, database_path: str):
        self._path = database_path
        with self._open() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    prompt TEXT,
                    winner TEXT NOT NULL,
                    rivals_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0,
                    is_exploration INTEGER NOT NULL DEFAULT 0,
                    requested_json TEXT,
                    actual_json TEXT,
                    feedback_type INTEGER,
                    feedback_score REAL)
            """)

    def append(self, trace: Tracert, requested: Specifications | None = None,
               actual: Specifications | None = None, prompt: str | None = None) -> int:
        """Записывает ход. Возвращает идентификатор, под которым позже проставляется отзыв."""
        if not trace.winner.name or not trace.winner.name.strip():
            raise ValueError("Победитель без имени: по такому ходу кандидата потом не опознать.")
        rivals = [element.name for element in trace.top_k_elements if element is not trace.winner]
        with self._open() as connection:
            cursor = connection.execute(
                "INSERT INTO rounds (created_at, prompt, winner, rivals_json, features_json, score, "
                "is_exploration, requested_json, actual_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(), prompt, trace.winner.name,
                    json.dumps(rivals), json.dumps(trace.input_feature_vector.tolist()),
                    trace.score, 1 if trace.is_exploration else 0,
                    None if requested is None else json.dumps(requested.to_dict()),
                    None if actual is None else json.dumps(actual.to_dict()),
                ),
            )
            return int(cursor.lastrowid)

    def set_feedback(self, round_id: int, feedback: Feedback) -> None:
        with self._open() as connection:
            updated = connection.execute(
                "UPDATE rounds SET feedback_type = ?, feedback_score = ? WHERE id = ?",
                (int(feedback.ftype), feedback.score, round_id)).rowcount
        if updated == 0:
            raise ValueError(f"Хода {round_id} в базе нет: отзыв не к чему привязать.")

    def read_rated(self, catalog: Iterable[RoutedElement], limit: int = 1000) -> list[TrainingRound]:
        """Обучающая выборка: ходы с отзывом, свежие первыми. Ход пропускается, если кто-то из
        участников больше не значится в каталоге: снятую модель незачем ни поощрять, ни
        наказывать."""
        by_name = {element.name: element for element in catalog if element.name}
        with self._open() as connection:
            rows = connection.execute(
                "SELECT id, winner, rivals_json, features_json, score, requested_json, actual_json, "
                "feedback_type, feedback_score, is_exploration FROM rounds "
                "WHERE feedback_score IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        rounds = []
        for row in rows:
            winner = by_name.get(row[1])
            rival_names = json.loads(row[2])
            if winner is None or any(name not in by_name for name in rival_names):
                continue
            trace = Tracert(
                winner=winner,
                top_k_elements=[winner, *(by_name[name] for name in rival_names)],
                input_feature_vector=np.array(json.loads(row[3])),
                score=row[4],
                is_exploration=bool(row[9]),
            )
            feedback = Feedback(FeedbackType(row[7]), row[8])
            rounds.append(TrainingRound(row[0], trace, feedback, _spec_of(row[5]), _spec_of(row[6])))
        return rounds

    def load_statistics(self, elements: Iterable[RoutedElement]) -> int:
        """Опыт кандидатов из журнала: число оцененных ходов и оценка дисперсии отзывов. Эти
        величины нужны формуле температуры, и копить их отдельно не требуется."""
        by_name = {element.name: element for element in elements if element.name}
        with self._open() as connection:
            rows = connection.execute(
                "SELECT winner, COUNT(*), AVG(feedback_score), AVG(feedback_score * feedback_score) "
                "FROM rounds WHERE feedback_score IS NOT NULL GROUP BY winner").fetchall()
        restored = 0
        for name, count, mean, mean_of_squares in rows:
            element = by_name.get(name)
            if element is None:
                continue
            element.experience = count
            # Поправка на несмещенность: по одному ходу дисперсию не оценить
            element.score_variance = 0.0 if count < 2 else max(
                0.0, (mean_of_squares - mean * mean) * count / (count - 1))
            restored += 1
        return restored

    def get_feature_mean(self) -> np.ndarray | None:
        """Среднее по векторам задач, для Settings.task_mean. Ходы хранятся несмещенными ради
        этого расчета: иначе среднее считалось бы само из себя и уползало."""
        with self._open() as connection:
            rows = connection.execute("SELECT features_json FROM rounds").fetchall()
        if not rows:
            return None
        return np.mean([json.loads(row[0]) for row in rows], axis=0)

    def count(self) -> tuple[int, int]:
        """Сколько ходов записано и сколько из них оценено."""
        with self._open() as connection:
            row = connection.execute("SELECT COUNT(*), COUNT(feedback_score) FROM rounds").fetchone()
        return int(row[0]), int(row[1])

    def _open(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)


def _spec_of(raw: str | None) -> Specifications | None:
    return None if raw is None else Specifications.from_dict(json.loads(raw))
