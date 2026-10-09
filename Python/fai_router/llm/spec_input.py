from __future__ import annotations

import json
import math
from typing import Any

from fai_router.enums import Domain, ProgrammingLanguage, ScienceField, Style, TaskKind
from fai_router.llm import field_descriptions, json_call, prompt_data
from fai_router.settings import Settings
from fai_router.specifications import Specifications, enum_value


class SpecInputRecognizer:
    """Извлечение задания из текста запроса одним обращением к модели (при негодном ответе еще одним).

    Запрос пользователя идет модели внутри метки со случайным именем (prompt_data): ограду из
    черточек закрывала та же строка черточек в запросе. Ответ проверяется: значение вне перечисления
    (в том числе число вместо имени), пустые пункты и неизвестные имена полей отбрасываются, а негодный
    ответ повторяется один раз и затем это сбой распознавания. Тексты общие с версией на C#."""

    SYSTEM_PROMPT = (
        "Извлеки техническое задание из запроса пользователя к языковой модели.\n"
        "Опиши, каким должен быть ОТВЕТ на этот запрос, и верни параметры ответа в JSON по схеме.\n"
        "Явно заданные требования (объем, стиль, число разделов, таблицы, язык) бери как есть и перечисли "
        "имена этих полей в explicitFields.\n"
        "Неуказанное оценивай разумным ожиданием для такой задачи, а не нулем, но в explicitFields не вноси: "
        "угаданное требованием не считается.\n"
        "Язык ответа задан явно, если он назван или если запрос написан на нем и другой язык не назван.\n"
        "Отдельно выпиши смысловые пункты, которые ответ обязан раскрыть, и явные ограничения запроса.\n"
        "Область определяй по функции, которой служит результат: работа с покупателем и продажи "
        "(коммерческое предложение, письмо клиенту о ценах, скрипт звонка) это Sales, продвижение и "
        "реклама это Marketing, персонал это Hr, склад, закупки и возвраты это Operations, налоги и "
        "отчетность это Finance. Business ставь только для стратегии и управления компанией в целом.\n"
    )

    # Бюджет времени на распознавание, все попытки вместе: ход ждет его до выбора исполнителя
    BUDGET = 60.0

    # Поля схемы и поля Specifications: целые, дробные, логическое и строки
    _INTEGERS = {
        "symbolLength": "symbol_length", "wordLength": "word_length", "paragraphCount": "paragraph_count",
        "sectionCount": "section_count", "listItemCount": "list_item_count", "tableCount": "table_count",
        "codeBlockCount": "code_block_count", "formulaCount": "formula_count", "headingDepth": "heading_depth",
    }
    _NUMBERS = {
        "avgSentenceLength": "avg_sentence_length", "readabilityScore": "readability_score",
        "termDensity": "term_density", "formalityScore": "formality_score", "expertLevel": "expert_level",
        "difficulty": "difficulty", "factualityDemand": "factuality_demand",
    }
    _ENUMS = {
        "styleType": ("style_type", Style), "domain": ("domain", Domain),
        "programmingLanguage": ("programming_language", ProgrammingLanguage),
        "scienceField": ("science_field", ScienceField), "taskKind": ("task_kind", TaskKind),
    }

    SCHEMA = {
        "type": "object",
        "properties": {
            "styleType": {"type": "string", "enum": [style.value for style in Style],
                          "description": "Стиль текста ответа"},
            "symbolLength": {"type": "integer", "description": "Объем ответа в символах"},
            "wordLength": {"type": "integer", "description": "Объем ответа в словах"},
            "paragraphCount": {"type": "integer", "description": "Число абзацев"},
            "sectionCount": {"type": "integer", "description": "Число разделов"},
            "listItemCount": {"type": "integer", "description": "Число пунктов списков"},
            "tableCount": {"type": "integer", "description": "Число таблиц"},
            "codeBlockCount": {"type": "integer", "description": "Число блоков кода"},
            "formulaCount": {"type": "integer", "description": "Число формул"},
            "headingDepth": {"type": "integer", "description": "Глубина вложенности заголовков"},
            "avgSentenceLength": {"type": "number", "description": "Средняя длина предложения в словах"},
            "readabilityScore": {"type": "number", "minimum": 0, "maximum": 100,
                                 "description": field_descriptions.READABILITY},
            "termDensity": {"type": "number", "minimum": 0, "maximum": 1,
                            "description": field_descriptions.TERM_DENSITY},
            "formalityScore": {"type": "number", "minimum": 0, "maximum": 1,
                               "description": field_descriptions.FORMALITY},
            "language": {"type": "string", "enum": [*Specifications.LANGUAGE_CODES, "other"],
                         "description": field_descriptions.LANGUAGE},
            "hasReferences": {"type": "boolean", "description": "Нужны ли ссылки на источники"},
            "domain": {"type": "string", "enum": [item.value for item in Domain],
                       "description": field_descriptions.DOMAIN},
            "programmingLanguage": {"type": "string", "enum": [item.value for item in ProgrammingLanguage],
                                    "description": field_descriptions.PROGRAMMING_LANGUAGE},
            "scienceField": {"type": "string", "enum": [item.value for item in ScienceField],
                             "description": field_descriptions.SCIENCE_FIELD},
            "taskKind": {"type": "string", "enum": [item.value for item in TaskKind],
                         "description": field_descriptions.TASK_KIND},
            "expertLevel": {"type": "number", "minimum": 0, "maximum": 1,
                            "description": field_descriptions.EXPERT_LEVEL},
            "difficulty": {"type": "number", "minimum": 0, "maximum": 1,
                           "description": field_descriptions.DIFFICULTY},
            "factualityDemand": {"type": "number", "minimum": 0, "maximum": 1,
                                 "description": field_descriptions.FACTUALITY_DEMAND},
            "requiredPoints": {"type": "array", "items": {"type": "string"},
                               "description": field_descriptions.REQUIRED_POINTS},
            "constraints": {"type": "array", "items": {"type": "string"},
                            "description": field_descriptions.CONSTRAINTS},
            "explicitFields": {"type": "array",
                               "items": {"type": "string", "enum": list(Specifications.STATABLE_FIELDS)},
                               "description": field_descriptions.EXPLICIT_FIELDS},
        },
        "required": ["styleType", "symbolLength", "wordLength", "paragraphCount", "sectionCount",
                     "listItemCount", "tableCount", "codeBlockCount", "formulaCount", "headingDepth",
                     "avgSentenceLength", "readabilityScore", "termDensity", "formalityScore",
                     "language", "hasReferences", "domain", "programmingLanguage", "scienceField", "taskKind",
                     "expertLevel", "difficulty", "factualityDemand", "requiredPoints", "constraints",
                     "explicitFields"],
        "additionalProperties": False,
    }

    def __init__(self, llm=None, budget: float = BUDGET):
        self._llm = llm
        self.budget = budget

    def get_specifications(self, prompt: str) -> Specifications:
        """Ожидаемая спецификация ответа по тексту запроса. Модель дважды ответила не по схеме,
        тогда json_call.InvalidModelAnswer; не уложилась в бюджет, тогда TimeoutError."""
        if not prompt or not prompt.strip():
            raise ValueError("Запрос не может быть пустым.")
        tag = prompt_data.new_tag()
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT + "\n" + prompt_data.rule(tag)},
                    {"role": "user", "content": prompt_data.wrap(tag, "запрос пользователя", prompt)}]
        return json_call.ask(self._llm or Settings.require_llm(), messages, "input_specifications",
                             self.SCHEMA, self.read, self.budget)

    @classmethod
    def read(cls, raw: str) -> Specifications | None:
        """Разбор ответа модели: значения перечислений только из перечислений и только именами,
        числа только числами, пункты без пустых и повторов и не больше MAX_ITEMS, язык кодом без
        региона, явные поля только из STATABLE_FIELDS. None, если ответ негоден. Пропущенное поле
        остается по умолчанию."""
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        spec = Specifications()
        for key, (name, enum_cls) in cls._ENUMS.items():
            if key in data:
                value = enum_value(enum_cls, data[key])
                if value is None:
                    return None
                setattr(spec, name, value)
        for keys, cast in ((cls._INTEGERS, int), (cls._NUMBERS, float)):
            for key, name in keys.items():
                if key in data:
                    number = _number(data[key])
                    if number is None or (cast is int and number != int(number)):
                        return None
                    setattr(spec, name, cast(number))
        if "hasReferences" in data:
            if not isinstance(data["hasReferences"], bool):
                return None
            spec.has_references = data["hasReferences"]
        language = data.get("language")
        spec.language = Specifications.normalize_language(language) if isinstance(language, str) else None
        spec.required_points = items(data.get("requiredPoints"))
        spec.constraints = items(data.get("constraints"))
        explicit = data.get("explicitFields")
        if isinstance(explicit, list):
            named = {str(field).lower() for field in explicit}
            spec.explicit_fields = [field for field in Specifications.STATABLE_FIELDS if field.lower() in named]
        return spec


def items(values: Any) -> list[str]:
    """Пункты без пустых и повторов, не больше предела: null в списке и сам список null модель
    присылала, а дальше по ним шел суд."""
    cleaned: list[str] = []
    for value in values if isinstance(values, list) else []:
        text = value.strip() if isinstance(value, str) else ""
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:Specifications.MAX_ITEMS]


def _number(value: Any) -> float | None:
    """Конечное число из ответа модели; строка, логическое и нечисловое дают None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)
