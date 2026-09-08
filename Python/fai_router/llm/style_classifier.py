from __future__ import annotations

import json
from dataclasses import dataclass

from fai_router.enums import Style
from fai_router.llm import field_descriptions
from fai_router.llm.client import OpenRouterClient
from fai_router.settings import Settings


@dataclass(frozen=True)
class StyleAssessment:
    """Смысловые признаки текста, которые распознает модель."""

    style_type: Style = Style.OTHER
    term_density: float = 0.0
    formality_score: float = 0.0


class StyleClassifier:
    """Смысловая оценка текста моделью: стиль и лексические метрики, то есть все, что не
    считается по разметке."""

    SYSTEM_PROMPT = (
        "Ты оцениваешь стиль и лексику присланного текста. Определи стиль, долю терминологии "
        "и формальность тона, верни результат строго в виде JSON по заданной схеме, без пояснений."
    )

    SCHEMA = {
        "type": "object",
        "properties": {
            "styleType": {"type": "string", "enum": [style.value for style in Style],
                          "description": "Стиль текста"},
            "termDensity": {"type": "number", "minimum": 0, "maximum": 1,
                            "description": field_descriptions.TERM_DENSITY},
            "formalityScore": {"type": "number", "minimum": 0, "maximum": 1,
                               "description": field_descriptions.FORMALITY},
        },
        "required": ["styleType", "termDensity", "formalityScore"],
        "additionalProperties": False,
    }

    def __init__(self, llm: OpenRouterClient | None = None):
        # Свой клиент нужен, когда в одном процессе судят несколько моделей
        self._llm = llm

    def assess(self, text: str) -> StyleAssessment:
        if not text or not text.strip():
            raise ValueError("Текст для оценки не может быть пустым.")
        client = self._llm or Settings.require_llm()
        raw = client.complete(
            [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": text}],
            schema=self.SCHEMA, schema_name="style_assessment",
        )
        data = json.loads(raw)
        return StyleAssessment(
            style_type=_style_of(data.get("styleType")),
            term_density=float(data.get("termDensity", 0.0)),
            formality_score=float(data.get("formalityScore", 0.0)),
        )


def _style_of(value: str | None) -> Style:
    try:
        return Style(value)
    except ValueError:
        return Style.OTHER
