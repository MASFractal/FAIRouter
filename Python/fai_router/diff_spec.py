from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator

from fai_router import content_review as cr
from fai_router.content_review import ContentReview
from fai_router.specifications import Specifications


@dataclass(frozen=True)
class SpecDeviation:
    """Расхождение по одному пункту задания. Отклонение: ноль означает совпадение, единица
    означает полное расхождение. content отличает пункт содержания от пункта формы."""

    field: str
    requested: str
    actual: str
    deviation: float
    content: bool = False


class DiffSpec:
    """Разбор всех расхождений между заданием и фактом по каждому пункту (режим критика).

    Форма сверяется всегда: 20 пунктов от стиля и объема до типа задачи. С оценкой содержания
    разбор включает и его: каждый смысловой пункт заказа, каждое ограничение, экспертность ответа
    против заказанной, каждое проверяемое утверждение, расчеты, наполнение структуры, источники и
    пригодность для дела. Трудности и длины диалога в разборе нет: это свойства запроса, у ответа им
    нечего противопоставить."""

    # Отклонение, начиная с которого пункт считается проваленным
    MISMATCH_THRESHOLD = 0.2

    def __init__(self, deviations: list[SpecDeviation]):
        self.deviations = deviations

    @property
    def total_deviation(self) -> float:
        """Среднее отклонение по всем пунктам, формы и содержания."""
        return sum(item.deviation for item in self.deviations) / len(self.deviations)

    @property
    def form_deviation(self) -> float:
        """Среднее отклонение по пунктам формы: итог формы равен единице минус это число."""
        form = [item.deviation for item in self.deviations if not item.content]
        return sum(form) / len(form)

    @property
    def mismatches(self) -> list[SpecDeviation]:
        """Проваленные пункты, худшие первыми."""
        failed = [item for item in self.deviations if item.deviation > self.MISMATCH_THRESHOLD]
        return sorted(failed, key=lambda item: item.deviation, reverse=True)

    @classmethod
    def compare(cls, requested: Specifications, actual: Specifications,
                content: ContentReview | None = None) -> "DiffSpec":
        form = cls._form(requested, actual)
        return cls(form if content is None else form + list(cls._content(requested, content)))

    def __str__(self) -> str:
        """Отчет критика: список проваленных пунктов."""
        return "\n".join(
            f"{item.field}: заказано {item.requested}, получено {item.actual}"
            for item in self.mismatches
        )

    @classmethod
    def _form(cls, requested: Specifications, actual: Specifications) -> list[SpecDeviation]:
        number, exact = cls._number, cls._exact
        return [
            exact("Стиль", requested.style_type.value, actual.style_type.value),
            number("Объем в символах", requested.symbol_length, actual.symbol_length),
            number("Объем в словах", requested.word_length, actual.word_length),
            number("Абзацы", requested.paragraph_count, actual.paragraph_count),
            number("Разделы", requested.section_count, actual.section_count),
            number("Пункты списков", requested.list_item_count, actual.list_item_count),
            number("Таблицы", requested.table_count, actual.table_count),
            number("Блоки кода", requested.code_block_count, actual.code_block_count),
            number("Формулы", requested.formula_count, actual.formula_count),
            number("Глубина заголовков", requested.heading_depth, actual.heading_depth),
            number("Средняя длина предложения", requested.avg_sentence_length, actual.avg_sentence_length),
            number("Читаемость", requested.readability_score, actual.readability_score),
            number("Доля терминологии", requested.term_density, actual.term_density),
            number("Формальность", requested.formality_score, actual.formality_score),
            exact("Язык", requested.language, actual.language),
            exact("Ссылки на источники", requested.has_references, actual.has_references),
            exact("Область", requested.domain.value, actual.domain.value),
            exact("Язык программирования", requested.programming_language.value, actual.programming_language.value),
            exact("Область науки", requested.science_field.value, actual.science_field.value),
            exact("Тип задачи", requested.task_kind.value, actual.task_kind.value),
        ]

    @classmethod
    def _content(cls, requested: Specifications, content: ContentReview) -> Iterator[SpecDeviation]:
        """Пункты содержания: у каждого смыслового пункта и ограничения своя строка, у каждого
        проверяемого утверждения тоже. Общая оценка критерия идет строкой, только когда поштучного
        разбора нет: иначе одно расхождение попало бы в разбор дважды."""
        for point in content.points:
            yield SpecDeviation(f"Смысловой пункт «{point.point}»", "раскрыть", _coverage_text(point.coverage),
                                1.0 - point.coverage, True)
        if not content.points:
            yield cls._criterion(content, cr.COMPLETENESS)
        for check in content.constraint_checks:
            yield SpecDeviation(f"Ограничение «{check.constraint}»", "соблюсти",
                                "соблюдено" if check.met else "нарушено", 0.0 if check.met else 1.0, True)
        if requested.expert_level > 0 and content.expert_level is not None:
            yield replace(cls._number("Экспертность", requested.expert_level, content.expert_level), content=True)
        else:
            yield cls._criterion(content, cr.EXPERTISE)
        for claim in content.claims:
            yield SpecDeviation(f"Факт «{claim.text}»", "верно", f"верно с вероятностью {claim.truth:.2f}",
                                1.0 - claim.truth, True)
        for name in (cr.REASONING, cr.STRUCTURE_CONTENT, cr.SOURCE_QUALITY, cr.FIT_FOR_PURPOSE):
            if content.get(name) is not None:
                yield cls._criterion(content, name)

    @staticmethod
    def _criterion(content: ContentReview, name: str) -> SpecDeviation:
        score = content.get(name)
        score = 1.0 if score is None else score
        return SpecDeviation(name, "1", f"{score:.4g}", 1.0 - score, True)

    @staticmethod
    def _number(field: str, requested: float, actual: float) -> SpecDeviation:
        # Заказан ноль, а получено больше нуля: расхождение полное, делить не на что
        if requested == 0:
            deviation = 0.0 if actual == 0 else 1.0
        else:
            deviation = min(1.0, abs(actual - requested) / abs(requested))
        return SpecDeviation(field, f"{requested:.4g}", f"{actual:.4g}", deviation)

    @staticmethod
    def _exact(field: str, requested: Any, actual: Any) -> SpecDeviation:
        return SpecDeviation(
            field,
            "нет" if requested is None else str(requested),
            "нет" if actual is None else str(actual),
            0.0 if requested == actual else 1.0,
        )


def _coverage_text(coverage: float) -> str:
    return "раскрыт" if coverage >= 0.8 else "раскрыт частично" if coverage >= 0.3 else "не раскрыт"
