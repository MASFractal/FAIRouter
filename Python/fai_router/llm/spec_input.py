from __future__ import annotations

import json

from fai_router.enums import Domain, ProgrammingLanguage, ScienceField, Style, TaskKind
from fai_router.llm import field_descriptions
from fai_router.llm.client import OpenRouterClient
from fai_router.settings import Settings
from fai_router.specifications import Specifications


class SpecInputRecognizer:
    """Извлечение задания из текста запроса одним обращением к модели."""

    PROMPT = (
        "Извлеки техническое задание из запроса пользователя к языковой модели.\n"
        "Опиши, каким должен быть ОТВЕТ на этот запрос, и верни параметры ответа в JSON по схеме.\n"
        "Явно заданные требования (объем, стиль, число разделов, таблицы, язык) бери как есть.\n"
        "Неуказанное оценивай разумным ожиданием для такой задачи, а не нулем.\n"
        "Отдельно выпиши смысловые пункты, которые ответ обязан раскрыть, и явные ограничения запроса.\n\n"
        "Запрос пользователя:\n----\n{text}\n----"
    )

    # Имена полей схемы отображаются на поля Specifications таблицей ниже; поля-перечисления
    # разбираются по своей таблице, незнакомое значение остается «не задано»
    _ENUMS = {
        "styleType": ("style_type", Style),
        "domain": ("domain", Domain),
        "programmingLanguage": ("programming_language", ProgrammingLanguage),
        "scienceField": ("science_field", ScienceField),
        "taskKind": ("task_kind", TaskKind),
    }

    _KEYS = {
        "styleType": None,
        "symbolLength": "symbol_length",
        "wordLength": "word_length",
        "paragraphCount": "paragraph_count",
        "sectionCount": "section_count",
        "listItemCount": "list_item_count",
        "tableCount": "table_count",
        "codeBlockCount": "code_block_count",
        "formulaCount": "formula_count",
        "headingDepth": "heading_depth",
        "avgSentenceLength": "avg_sentence_length",
        "readabilityScore": "readability_score",
        "termDensity": "term_density",
        "formalityScore": "formality_score",
        "language": "language",
        "hasReferences": "has_references",
        "domain": None,
        "programmingLanguage": None,
        "scienceField": None,
        "taskKind": None,
        "expertLevel": "expert_level",
        "difficulty": "difficulty",
        "factualityDemand": "factuality_demand",
        "requiredPoints": "required_points",
        "constraints": "constraints",
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
                                 "description": "Читаемость по Флешу-Кинкейду, 0-100"},
            "termDensity": {"type": "number", "minimum": 0, "maximum": 1,
                            "description": field_descriptions.TERM_DENSITY},
            "formalityScore": {"type": "number", "minimum": 0, "maximum": 1,
                               "description": field_descriptions.FORMALITY},
            "language": {"type": "string", "description": "Язык ответа, код ISO 639-1"},
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
        },
        "required": list(_KEYS),
        "additionalProperties": False,
    }

    def __init__(self, llm: OpenRouterClient | None = None):
        self._llm = llm

    def get_specifications(self, prompt: str) -> Specifications:
        if not prompt or not prompt.strip():
            raise ValueError("Запрос не может быть пустым.")
        client = self._llm or Settings.require_llm()
        raw = client.complete(
            [{"role": "user", "content": self.PROMPT.format(text=prompt)}],
            schema=self.SCHEMA, schema_name="input_specifications",
        )
        data = json.loads(raw)
        spec = Specifications()
        for key, attribute in self._KEYS.items():
            if key not in data:
                continue
            if attribute is None:
                name, enum_cls = self._ENUMS[key]
                try:
                    setattr(spec, name, enum_cls(data[key]))
                except ValueError:
                    pass
            else:
                setattr(spec, attribute, data[key])
        return spec
