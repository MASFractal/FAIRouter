from __future__ import annotations

import math
from typing import Any

import numpy as np

from fai_router.enums import Style


class Specifications:
    """Спецификация (техническое задание). Один и тот же тип описывает и требования к ответу,
    и свойства готового текста, поэтому заказ и факт можно сравнить."""

    # Типичный масштаб поля. Вектор сравнивается косинусом, поэтому координаты обязаны быть
    # соизмеримы: в сырых единицах объем в символах давал 98% длины вектора, и косинус мерил
    # только длину, из-за чего текст в противоположном стиле получал оценку 0,9998
    SYMBOL_LENGTH_SCALE = 20000.0
    WORD_LENGTH_SCALE = 3000.0
    COUNT_SCALE = 20.0
    HEADING_DEPTH_SCALE = 6.0
    SENTENCE_LENGTH_SCALE = 40.0
    READABILITY_MAX = 100.0

    def __init__(
        self,
        style_type: Style = Style.OTHER,
        symbol_length: int = 0,
        word_length: int = 0,
        paragraph_count: int = 0,
        section_count: int = 0,
        list_item_count: int = 0,
        table_count: int = 0,
        code_block_count: int = 0,
        formula_count: int = 0,
        heading_depth: int = 0,
        avg_sentence_length: float = 0.0,
        readability_score: float = 0.0,
        term_density: float = 0.0,
        formality_score: float = 0.0,
        language: str | None = None,
        has_references: bool = False,
    ):
        self.style_type = style_type
        self.symbol_length = symbol_length
        self.word_length = word_length
        self.paragraph_count = paragraph_count
        self.section_count = section_count
        self.list_item_count = list_item_count
        self.table_count = table_count
        self.code_block_count = code_block_count
        self.formula_count = formula_count
        self.heading_depth = heading_depth
        self.avg_sentence_length = avg_sentence_length
        self.readability_score = readability_score
        self.term_density = term_density
        self.formality_score = formality_score
        self.language = language
        self.has_references = has_references

    # Доли приходят от модели, а она границы схемы соблюдает не всегда: DeepSeek возвращал
    # term_density 4 и 80 при объявленных 0..1. Такое значение забивает длину вектора целиком,
    # поэтому границу держит сам тип, а не только схема ответа.

    @property
    def readability_score(self) -> float:
        return self._readability

    @readability_score.setter
    def readability_score(self, value: float) -> None:
        self._readability = min(max(float(value), 0.0), self.READABILITY_MAX)

    @property
    def term_density(self) -> float:
        return self._term_density

    @term_density.setter
    def term_density(self, value: float) -> None:
        self._term_density = min(max(float(value), 0.0), 1.0)

    @property
    def formality_score(self) -> float:
        return self._formality

    @formality_score.setter
    def formality_score(self, value: float) -> None:
        self._formality = min(max(float(value), 0.0), 1.0)

    def feature_vector(self) -> np.ndarray:
        """Вектор признаков: код стиля «один из многих» и приведенные к масштабу метрики.
        Язык и наличие ссылок в вектор не входят: способ их кодирования не определен."""
        style = np.zeros(len(Style))
        style[self.style_type.index] = 1.0
        scaled = self.scaled
        numeric = np.array([
            scaled(self.symbol_length, self.SYMBOL_LENGTH_SCALE),
            scaled(self.word_length, self.WORD_LENGTH_SCALE),
            scaled(self.paragraph_count, self.COUNT_SCALE),
            scaled(self.section_count, self.COUNT_SCALE),
            scaled(self.list_item_count, self.COUNT_SCALE),
            scaled(self.table_count, self.COUNT_SCALE),
            scaled(self.code_block_count, self.COUNT_SCALE),
            scaled(self.formula_count, self.COUNT_SCALE),
            scaled(self.heading_depth, self.HEADING_DEPTH_SCALE),
            scaled(self.avg_sentence_length, self.SENTENCE_LENGTH_SCALE),
            self.readability_score / self.READABILITY_MAX,
            self.term_density,
            self.formality_score,
        ])
        return np.concatenate([style, numeric])

    @staticmethod
    def scaled(value: float, scale: float) -> float:
        """Логарифмическая шкала: у объемов и счетчиков значимо отношение величин, а не
        разница, а деление на масштаб приводит поле к единичному порядку. Отрицательное
        значение может прийти от модели, а логарифм на нем дает NaN и портит весь вектор."""
        return math.log(1 + max(0.0, float(value))) / math.log(1 + scale)

    _FIELDS = (
        "symbol_length", "word_length", "paragraph_count", "section_count",
        "list_item_count", "table_count", "code_block_count", "formula_count",
        "heading_depth", "avg_sentence_length", "readability_score", "term_density",
        "formality_score", "language", "has_references",
    )

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in self._FIELDS}
        data["style_type"] = self.style_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Specifications":
        spec = cls()
        for name in cls._FIELDS:
            if name in data and data[name] is not None:
                setattr(spec, name, data[name])
        style = data.get("style_type")
        if style is not None:
            spec.style_type = Style(style)
        return spec
