"""Оценка содержания ответа: то, что не видно по форме.

Таблица с выдуманными цифрами, отчет не про ту аналитику и несуществующие источники проходят
сверку формы на отлично, а здесь проваливаются. Критерий, который к задаче не относится, в среднее
не входит: у задачи без ограничений нечего выполнять, у ответа без проверяемых утверждений нечего
проверять. Воздержание от фактов не штрафуется, ложный факт штрафуется, как в рейтинге фактологии
арены."""

from __future__ import annotations

from dataclasses import dataclass, field

FACTUALITY = "Фактология"
COMPLETENESS = "Полнота по сути"
INSTRUCTION_FOLLOWING = "Выполнение указаний"
REASONING = "Верность рассуждений и расчетов"
EXPERTISE = "Экспертная глубина"
STRUCTURE_CONTENT = "Наполнение структуры"
SOURCE_QUALITY = "Качество источников"
FIT_FOR_PURPOSE = "Пригодность для дела"

# Оценка, ниже которой критерий считается проваленным
PASS_MARK = 0.6


@dataclass(frozen=True)
class FactClaim:
    """Проверяемое утверждение ответа и вероятность, что оно верно."""

    text: str
    truth: float


@dataclass(frozen=True)
class ContentCriterion:
    """Оценка одного критерия; None, если к этой задаче критерий не относится."""

    name: str
    score: float | None


@dataclass
class ContentReview:
    """Оценка содержания по критериям, проверяемые утверждения и замечания."""

    criteria: list[ContentCriterion]
    claims: list[FactClaim] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Среднее по критериям, которые относятся к задаче."""
        scores = [item.score for item in self.criteria if item.score is not None]
        return sum(scores) / len(scores) if scores else 1.0

    def get(self, name: str) -> float | None:
        return next((item.score for item in self.criteria if item.name == name), None)

    @staticmethod
    def factuality_of(claims: list[FactClaim]) -> float | None:
        """Средняя вероятность истинности; утверждений нет, значит None."""
        if not claims:
            return None
        return sum(min(max(claim.truth, 0.0), 1.0) for claim in claims) / len(claims)

    def __str__(self) -> str:
        lines = [f"Содержание {self.score:.2f}"]
        lines += [f"  {item.name}: {item.score:.2f}" for item in self.criteria
                  if item.score is not None and item.score < PASS_MARK]
        lines += [f"  Сомнительное утверждение ({claim.truth:.2f}): {claim.text}"
                  for claim in self.claims if claim.truth < 0.5]
        lines += [f"  - {issue}" for issue in self.issues]
        return "\n".join(lines)
