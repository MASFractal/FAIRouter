from __future__ import annotations

import math
from typing import Any

import numpy as np

from fai_router.enums import Domain, Indexed, ProgrammingLanguage, ScienceField, Style, TaskKind


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

    # Языки, у которых на арене свой рейтинг, кодами ISO 639-1. Язык из списка светит своим
    # разрядом, любой другой известный язык светит последним, неизвестный не светит ничем
    LANGUAGE_CODES = ("en", "ru", "zh", "fr", "de", "es", "ja", "ko", "pl")

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
        domain: Domain = Domain.GENERAL,
        programming_language: ProgrammingLanguage = ProgrammingLanguage.NONE,
        science_field: ScienceField = ScienceField.NONE,
        task_kind: TaskKind = TaskKind.NONE,
        expert_level: float = 0.0,
        difficulty: float = 0.0,
        factuality_demand: float = 0.0,
        required_points: list[str] | None = None,
        constraints: list[str] | None = None,
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
        # Предмет задачи: по нему роутер учит, кто в какой области силен
        self.domain = domain
        self.programming_language = programming_language
        self.science_field = science_field
        self.task_kind = task_kind
        # Требования заказа вне вектора ответа: у готового текста их нет, поэтому в вектор
        # спецификации они не входят, а читает их InputFeatures как свойства задачи
        self.expert_level = expert_level
        self.difficulty = difficulty
        self.factuality_demand = factuality_demand
        # Смысловые пункты и явные ограничения: по ним судья содержания проверяет полноту и
        # выполнение указаний, а не число разделов
        self.required_points = list(required_points or [])
        self.constraints = list(constraints or [])

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

    @property
    def expert_level(self) -> float:
        """Насколько запрос требует экспертной подготовки, 0..1 (категория арены Expert)."""
        return self._expert_level

    @expert_level.setter
    def expert_level(self, value: float) -> None:
        self._expert_level = min(max(float(value), 0.0), 1.0)

    @property
    def difficulty(self) -> float:
        """Доля из семи признаков трудного запроса арены, 0..1 (Hard Prompts)."""
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        self._difficulty = min(max(float(value), 0.0), 1.0)

    @property
    def factuality_demand(self) -> float:
        """Насколько ответ держится на проверяемых фактах, 0..1."""
        return self._factuality_demand

    @factuality_demand.setter
    def factuality_demand(self, value: float) -> None:
        self._factuality_demand = min(max(float(value), 0.0), 1.0)

    @classmethod
    def language_dim(cls) -> int:
        """Разрядов под язык: по одному на язык арены и один на прочие."""
        return len(cls.LANGUAGE_CODES) + 1

    @classmethod
    def language_slot(cls, code: str | None) -> int:
        """Разряд языка в векторе: по списку LANGUAGE_CODES, последний для прочих, минус единица
        для неизвестного."""
        if not code or not code.strip():
            return -1
        code = code.strip().lower()
        return cls.LANGUAGE_CODES.index(code) if code in cls.LANGUAGE_CODES else len(cls.LANGUAGE_CODES)

    def feature_vector(self) -> np.ndarray:
        """Вектор признаков: коды «один из многих» стиля и предмета задачи, приведенные к масштабу
        метрики, язык и признак ссылок. У предмета, языка и ссылок масштабов нет, как и у стиля."""
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
        language = np.zeros(self.language_dim())
        slot = self.language_slot(self.language)
        if slot >= 0:
            language[slot] = 1.0
        return np.concatenate([
            _one_hot(self.style_type), numeric,
            _subject(self.domain), _subject(self.programming_language),
            _subject(self.science_field), _subject(self.task_kind),
            language, [1.0 if self.has_references else 0.0],
        ])

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
        "expert_level", "difficulty", "factuality_demand",
    )

    _LIST_FIELDS = ("required_points", "constraints")

    # Поля-перечисления пишутся значениями, как в версии на C#; старые записи без них читаются
    # как «не задано»
    _ENUM_FIELDS = {
        "style_type": Style, "domain": Domain, "programming_language": ProgrammingLanguage,
        "science_field": ScienceField, "task_kind": TaskKind,
    }

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in self._FIELDS}
        for name in self._ENUM_FIELDS:
            data[name] = getattr(self, name).value
        for name in self._LIST_FIELDS:
            data[name] = list(getattr(self, name))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Specifications":
        spec = cls()
        for name in cls._FIELDS:
            if name in data and data[name] is not None:
                setattr(spec, name, data[name])
        for name, enum_cls in cls._ENUM_FIELDS.items():
            value = data.get(name)
            if value is not None:
                setattr(spec, name, enum_cls(value))
        for name in cls._LIST_FIELDS:
            setattr(spec, name, [str(item) for item in data.get(name) or []])
        return spec


def _one_hot(value: Indexed) -> np.ndarray:
    """Код «один из многих» по перечислению: разряд по порядку значения."""
    vector = np.zeros(len(type(value)))
    vector[value.index] = 1.0
    return vector


def _subject(value: Indexed) -> np.ndarray:
    """Код предмета задачи: как «один из многих», но первое значение означает «не задано» и
    разряда не имеет. Задача без предмета получает те же координаты, что и раньше, и оценки
    судьи на ней не меняются."""
    vector = np.zeros(len(type(value)) - 1)
    if value.index > 0:
        vector[value.index - 1] = 1.0
    return vector
