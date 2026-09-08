from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fai_router.specifications import Specifications


@dataclass(frozen=True)
class SpecDeviation:
    """Расхождение по одному пункту задания. Отклонение: ноль означает совпадение, единица
    означает полное расхождение."""

    field: str
    requested: str
    actual: str
    deviation: float


class DiffSpec:
    """Разбор расхождений между заданием и фактом по каждому пункту (режим критика).
    Пункты сравниваются по отдельности, без вектора и матрицы, поэтому разбор объясним
    для человека и не зависит от масштаба координат."""

    # Отклонение, начиная с которого пункт считается проваленным
    MISMATCH_THRESHOLD = 0.2

    def __init__(self, deviations: list[SpecDeviation]):
        self.deviations = deviations

    @property
    def total_deviation(self) -> float:
        return sum(item.deviation for item in self.deviations) / len(self.deviations)

    @property
    def mismatches(self) -> list[SpecDeviation]:
        """Проваленные пункты, худшие первыми."""
        failed = [item for item in self.deviations if item.deviation > self.MISMATCH_THRESHOLD]
        return sorted(failed, key=lambda item: item.deviation, reverse=True)

    @classmethod
    def compare(cls, requested: Specifications, actual: Specifications) -> "DiffSpec":
        number, exact = cls._number, cls._exact
        return cls([
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
        ])

    def __str__(self) -> str:
        """Отчет критика: список проваленных пунктов."""
        return "\n".join(
            f"{item.field}: заказано {item.requested}, получено {item.actual}"
            for item in self.mismatches
        )

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
