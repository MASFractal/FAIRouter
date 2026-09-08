from __future__ import annotations

import json

from fai_router.enums import Style
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
        "Неуказанное оценивай разумным ожиданием для такой задачи, а не нулем.\n\n"
        "Запрос пользователя:\n----\n{text}\n----"
    )

    # Имена полей схемы отображаются на поля Specifications таблицей ниже
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
                try:
                    spec.style_type = Style(data[key])
                except ValueError:
                    spec.style_type = Style.OTHER
            else:
                setattr(spec, attribute, data[key])
        return spec
