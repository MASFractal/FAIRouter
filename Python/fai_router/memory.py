"""Память роутера: журнал ходов, отзывы, обучение по ним, веса и калибровка планки. Устроена так
же, как RouterMemory в версии на C#.

Обучение и сохранение идут под одной блокировкой, а выбор их не ждет: обучение подменяет вектор
кандидата целиком, и соседний ход видит либо старый вектор, либо новый."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from fai_router.enums import FeedbackType
from fai_router.judge import Judge
from fai_router.persistence import SqliteTraceStore, SqliteWeightsStore
from fai_router.routed_element import RoutedElement
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert
from fai_router.training import Calibration, JudgeTrainer, RouterTrainer

# Сколько последних человеческих отзывов берет калибровка планки
CALIBRATION_LIMIT = 1000


class TrainingLoss(float):
    """Ошибка последней эпохи обучения, раздельно у роутера и у судьи: суммы штрафов по ходам
    эпохи, и сколько ходов вошло в обучение. Само значение это сумма обеих ошибок: так train
    возвращал ее до разделения, и прежний код, сравнивавший ее с числом, работает."""

    router: float
    judge: float
    rounds: int

    def __new__(cls, router: float, judge: float, rounds: int) -> "TrainingLoss":
        loss = super().__new__(cls, router + judge)
        loss.router, loss.judge, loss.rounds = router, judge, rounds
        return loss

    def __repr__(self) -> str:
        return f"TrainingLoss(router={self.router!r}, judge={self.judge!r}, rounds={self.rounds})"


@dataclass(frozen=True)
class CalibrationState:
    """Калибровка по журналу, доля лайков и число отзывов, на которых она подобрана."""

    fit: Calibration
    rate: float
    count: int


class RouterMemory:
    """Журнал и веса роутера в одной базе, обучение и калибровка планки."""

    def __init__(self, database_path: str, candidates: list[RoutedElement], judge: Judge):
        self._candidates = candidates
        self._judge = judge
        self._router_trainer = RouterTrainer(learning_rate=0.05)
        self._judge_trainer = JudgeTrainer(judge, learning_rate=0.5)
        self._gate = threading.RLock()
        # Кандидаты, чьи векторы обучены или загружены: сохраняются только они
        self._trained: set[str] = set()
        # Ходы, обученные после последнего save: отмечаются в журнале вместе с сохранением весов
        self._pending: set[int] = set()
        self._calibration: CalibrationState | None = None
        self.traces = SqliteTraceStore(database_path)
        self.weights = SqliteWeightsStore(database_path)

    @property
    def unsaved(self) -> bool:
        """Есть ли обученное, но не сохраненное: тогда при остановке стоит сохранить."""
        return bool(self._pending)

    def append(self, trace: Tracert, actual: Specifications | None, prompt: str, assessment: float | None) -> int:
        """Записывает ход и автоотзыв к нему; возвращает номер хода. Автоотзыв это итоговая оценка по
        содержанию и форме; человеческий, если придет, его перезапишет. Нечисловая или вне 0..1 оценка
        в обучение не идет: сбой судьи это отсутствие отзыва."""
        round_id = self.traces.append(trace, trace.requested_spec, actual, prompt)
        if assessment is not None and math.isfinite(assessment) and 0.0 <= assessment <= 1.0:
            self.traces.set_feedback(round_id, Feedback(FeedbackType.AUTO, assessment))
        return round_id

    def feedback(self, round_id: int, score: float, human: bool) -> None:
        """Отзыв к ходу; ход возвращается в очередь обучения, калибровка пересчитается."""
        self.traces.set_feedback(round_id, Feedback(FeedbackType.HUMAN if human else FeedbackType.AUTO, score))
        with self._gate:
            self._pending.discard(round_id)
        if human:
            self._calibration = None

    def current_calibration(self) -> CalibrationState:
        """Калибровка по журналу, запомненная до следующего человеческого отзыва, обучения или
        загрузки: прежде каждый ход читал тысячу строк журнала и заново решал задачу Ньютона."""
        state = self._calibration
        if state is None:
            pairs = self.calibration_pairs()
            state = self._calibration = (CalibrationState(Calibration(0.0, 0.0), 0.0, 0) if not pairs else
                                         CalibrationState(Calibration.fit(pairs),
                                                          sum(score for _, score in pairs) / len(pairs), len(pairs)))
        return state

    def calibration_pairs(self) -> list[tuple[float, float]]:
        return self.traces.read_calibration(self._candidates, CALIBRATION_LIMIT)

    def train(self, epochs: int) -> TrainingLoss:
        """Учит роутер и судью на ходах, которых еще не учили, в порядке записи."""
        with self._gate:
            sample = [item for item in self.traces.read_rated(self._candidates, untrained_only=True)
                      if item.id not in self._pending]
            router_loss = judge_loss = 0.0
            for _ in range(epochs):
                router_loss = judge_loss = 0.0
                for item in sample:
                    router_loss += self._router_trainer.train(item.trace, item.feedback)
                    self._trained.add(item.trace.winner.name)
                    # Судья учится только у человека. Автоотзыв это разбор расхождений по пунктам, и
                    # учить по нему судью значило бы подгонять одну автоматическую оценку под другую,
                    # а человек из этого круга выпадал бы совсем
                    if (item.feedback.ftype == FeedbackType.HUMAN
                            and item.requested is not None and item.actual is not None):
                        judge_loss += self._judge_trainer.train(item.requested, item.actual, item.feedback.score)
            self._pending.update(item.id for item in sample)
            self.traces.load_statistics(self._candidates)
            self._calibration = None
            return TrainingLoss(router_loss, judge_loss, len(sample))

    def save(self) -> None:
        """Сохраняет обученные векторы и судью одной транзакцией, затем отмечает обученные ходы."""
        with self._gate:
            self.weights.save([element for element in self._candidates if element.name in self._trained], self._judge)
            self.traces.mark_trained(self._pending)
            self._pending.clear()

    def load(self) -> None:
        """Загружает векторы со средним задач, судью и опыт кандидатов."""
        with self._gate:
            self._trained.update(self.weights.load_vectors(self._candidates))
            self.weights.load_judge(self._judge)
            self.traces.load_statistics(self._candidates)
            self._calibration = None
