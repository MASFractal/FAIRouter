from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from fai_router.enums import FeedbackType
from fai_router.persistence.db import connect, ensure_schema
from fai_router.routed_element import RoutedElement, by_name
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert

_ROUND_COLUMNS = ("id, winner, rivals_json, features_json, score, requested_json, actual_json, "
                  "feedback_type, feedback_score, is_exploration, predicted_quality")


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
    пишется отдельным действием.

    Журнал хранит тексты запросов. Срок их хранения задает владелец базы: purge удаляет старые ходы
    целиком либо только их тексты."""

    def __init__(self, database_path: str):
        self._path = database_path
        ensure_schema(database_path)

    def append(self, trace: Tracert, requested: Specifications | None = None,
               actual: Specifications | None = None, prompt: str | None = None) -> int:
        """Записывает ход. Возвращает идентификатор, под которым позже проставляется отзыв."""
        if not trace.winner.name or not trace.winner.name.strip():
            raise ValueError("Победитель без имени: по такому ходу кандидата потом не опознать.")
        rivals = [element.name or "" for element in trace.top_k_elements if element is not trace.winner]
        forecast = trace.forecast if trace.forecast is not None and math.isfinite(trace.forecast) else None
        with connect(self._path) as connection:
            cursor = connection.execute(
                "INSERT INTO rounds (created_at, prompt, winner, rivals_json, features_json, score, "
                "is_exploration, requested_json, actual_json, predicted_quality, failed_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _timestamp(datetime.now(timezone.utc)), prompt, trace.winner.name,
                    json.dumps(rivals, ensure_ascii=False), json.dumps(trace.input_feature_vector.tolist()),
                    trace.score, 1 if trace.is_exploration else 0,
                    None if requested is None else json.dumps(requested.to_dict(), ensure_ascii=False),
                    None if actual is None else json.dumps(actual.to_dict(), ensure_ascii=False),
                    forecast, json.dumps(trace.failed, ensure_ascii=False) if trace.failed else None,
                ),
            )
            return int(cursor.lastrowid)

    def set_feedback(self, round_id: int, feedback: Feedback) -> None:
        """Проставляет отзыв к записанному ходу. Ход снова попадает в очередь обучения: новый отзыв
        (человек поверх автоотзыва) должен переучить его своим знаком."""
        with connect(self._path) as connection:
            updated = connection.execute(
                "UPDATE rounds SET feedback_type = ?, feedback_score = ?, trained = 0 WHERE id = ?",
                (int(feedback.ftype), feedback.score, round_id)).rowcount
        if updated == 0:
            raise ValueError(f"Хода {round_id} в базе нет: отзыв не к чему привязать.")

    def read_rated(self, catalog: Iterable[RoutedElement], limit: int = 1000,
                   untrained_only: bool = False) -> list[TrainingRound]:
        """Обучающая выборка: последние limit ходов с отзывом, по порядку записи (старые первыми),
        чтобы последнее слово в обучении было за свежими. Победитель, снятый с каталога, выбрасывает
        ход: снятую модель незачем ни поощрять, ни наказывать. Снятый соперник выпадает только из пары.
        untrained_only оставляет ходы, которые еще не учили (mark_trained)."""
        named = by_name(catalog)
        untrained = "AND trained = 0" if untrained_only else ""
        with connect(self._path) as connection:
            rows = connection.execute(
                f"SELECT {_ROUND_COLUMNS} FROM (SELECT {_ROUND_COLUMNS} FROM rounds "
                f"WHERE feedback_score IS NOT NULL {untrained} ORDER BY id DESC LIMIT ?) ORDER BY id",
                (limit,)).fetchall()
        rounds = []
        for row in rows:
            winner = named.get(row[1])
            if winner is None:
                continue
            rivals = [named[name] for name in json.loads(row[2]) if name in named and named[name] is not winner]
            trace = Tracert(
                winner=winner,
                top_k_elements=[winner, *rivals],
                input_feature_vector=np.array(json.loads(row[3]), dtype=float),
                score=row[4],
                is_exploration=bool(row[9]),
                forecast=row[10],
            )
            rounds.append(TrainingRound(row[0], trace, Feedback(FeedbackType(row[7]), row[8]),
                                        _spec_of(row[5]), _spec_of(row[6])))
        return rounds

    def read_calibration(self, catalog: Iterable[RoutedElement], limit: int = 1000) -> list[tuple[float, float]]:
        """Пары для калибровки планки: прогноз качества победителя В МОМЕНТ ВЫБОРА и оценка человека,
        последние limit человеческих отзывов. Автоотзывы не берутся: планка обещает вероятность лайка
        человека, а не согласие судьи с самим собой.

        Прогноз при нынешних весах уже видел этот отзыв в обучении и обещал бы больше, чем знает.
        Ходы, записанные до появления колонки прогноза, берут прогноз нынешнего вектора победителя."""
        named = by_name(catalog)
        with connect(self._path) as connection:
            rows = connection.execute(
                "SELECT predicted_quality, winner, features_json, feedback_score FROM rounds "
                "WHERE feedback_type = ? AND feedback_score IS NOT NULL ORDER BY id DESC LIMIT ?",
                (int(FeedbackType.HUMAN), limit)).fetchall()
        pairs = []
        for forecast, winner_name, features, score in rows:
            if forecast is None and winner_name in named:
                forecast = named[winner_name].get_quality_score(np.array(json.loads(features), dtype=float))
            if forecast is not None and math.isfinite(forecast):
                pairs.append((float(forecast), float(score)))
        return pairs

    def mark_trained(self, round_ids: Iterable[int]) -> None:
        """Отмечает ходы обученными: следующее обучение их не повторит, пока к ним не придет новый отзыв."""
        with connect(self._path) as connection:
            connection.executemany("UPDATE rounds SET trained = 1 WHERE id = ?", [(round_id,) for round_id in round_ids])

    def load_statistics(self, elements: Iterable[RoutedElement]) -> int:
        """Опыт кандидатов из журнала: условное число оцененных ходов и оценка дисперсии отзывов. Эти
        величины нужны формуле температуры, и копить их отдельно не требуется.

        Человеческий отзыв идет в опыт целиком, автоотзыв с весом Settings.AUTO_FEEDBACK_WEIGHT.
        Прежде автоотзывы не считались вовсе, и без человеческих отзывов разведка не остывала никогда;
        еще раньше они считались наравне, и разведка гасла по мнению собственного судьи. Дисперсия
        стягивается к неизвестной (Settings.UNKNOWN_VARIANCE) с силой VARIANCE_PRIOR_STRENGTH: два
        одинаковых отзыва давали нулевую дисперсию и уверенность на пустом месте."""
        named = by_name(elements)
        with connect(self._path) as connection:
            rows = connection.execute(
                "SELECT winner, SUM(w), SUM(w * feedback_score), SUM(w * feedback_score * feedback_score) "
                "FROM (SELECT winner, feedback_score, CASE WHEN feedback_type = ? THEN 1.0 ELSE ? END AS w "
                "FROM rounds WHERE feedback_score IS NOT NULL) GROUP BY winner",
                (int(FeedbackType.HUMAN), Settings.AUTO_FEEDBACK_WEIGHT)).fetchall()
        restored = 0
        strength = Settings.VARIANCE_PRIOR_STRENGTH
        for name, weight, weighted, squares in rows:
            element = named.get(name)
            if element is None:
                continue
            mean = weighted / weight if weight > 0 else 0.0
            spread = max(0.0, squares / weight - mean * mean) if weight > 0 else 0.0
            element.experience = float(weight)
            element.score_variance = ((strength * Settings.UNKNOWN_VARIANCE + weight * spread)
                                      / (strength + max(weight - 1, 0.0)))
            restored += 1
        return restored

    def get_feature_mean(self, limit: int = 1000) -> np.ndarray | None:
        """Среднее по векторам задач последних limit ходов; None, если ходов еще нет. Ходы хранятся
        несмещенными ради этого расчета: иначе среднее считалось бы само из себя и уползало.
        Векторы другой длины (журнал прежней размерности признаков) пропускаются."""
        with connect(self._path) as connection:
            rows = connection.execute("SELECT features_json FROM rounds ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        vectors = [np.array(json.loads(row[0]), dtype=float) for row in rows]
        if not vectors:
            return None
        same = [vector for vector in vectors if len(vector) == len(vectors[0])]
        return np.mean(same, axis=0)

    def count(self) -> tuple[int, int]:
        """Сколько ходов записано и сколько из них оценено."""
        with connect(self._path) as connection:
            row = connection.execute("SELECT COUNT(*), COUNT(feedback_score) FROM rounds").fetchone()
        return int(row[0]), int(row[1])

    def purge(self, before: datetime, prompts_only: bool = False) -> int:
        """Срок хранения журнала: удаляет ходы, записанные раньше before, либо только стирает их
        тексты запросов, оставляя признаки и отзывы для обучения. Возвращает число ходов."""
        sql = ("UPDATE rounds SET prompt = NULL WHERE created_at < ? AND prompt IS NOT NULL" if prompts_only
               else "DELETE FROM rounds WHERE created_at < ?")
        moment = before if before.tzinfo is not None else before.replace(tzinfo=timezone.utc)
        with connect(self._path) as connection:
            return connection.execute(sql, (_timestamp(moment),)).rowcount


def _timestamp(moment: datetime) -> str:
    """Метка записи по UTC в том же виде, что пишет версия на C# (формат «O»): строки меток
    сравниваются в порядке времени."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "0Z"


def _spec_of(raw: str | None) -> Specifications | None:
    return None if raw is None else Specifications.from_dict(json.loads(raw))
