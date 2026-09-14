from __future__ import annotations

import json
from dataclasses import dataclass

from fai_router.enums import Domain, ScienceField, Style, TaskKind
from fai_router.llm import field_descriptions
from fai_router.llm.client import OpenRouterClient
from fai_router.settings import Settings


@dataclass(frozen=True)
class StyleAssessment:
    """Смысловые признаки текста, которые распознает модель."""

    style_type: Style = Style.OTHER
    term_density: float = 0.0
    formality_score: float = 0.0
    domain: Domain = Domain.GENERAL
    science_field: ScienceField = ScienceField.NONE
    task_kind: TaskKind = TaskKind.NONE


class StyleClassifier:
    """Смысловая оценка текста моделью: стиль и лексические метрики, то есть все, что не
    считается по разметке."""

    SYSTEM_PROMPT = (
        "Ты оцениваешь стиль, лексику и предмет присланного текста. Определи стиль, долю терминологии, "
        "формальность тона, предметную область, область науки и тип результата (что это за текст), верни результат "
        "строго в виде JSON по заданной схеме, без пояснений."
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
            "domain": {"type": "string", "enum": [item.value for item in Domain],
                       "description": field_descriptions.DOMAIN},
            "scienceField": {"type": "string", "enum": [item.value for item in ScienceField],
                             "description": field_descriptions.SCIENCE_FIELD},
            "taskKind": {"type": "string", "enum": [item.value for item in TaskKind],
                         "description": field_descriptions.TASK_KIND},
        },
        "required": ["styleType", "termDensity", "formalityScore", "domain", "scienceField", "taskKind"],
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
            style_type=_enum_of(Style, data.get("styleType"), Style.OTHER),
            term_density=float(data.get("termDensity", 0.0)),
            formality_score=float(data.get("formalityScore", 0.0)),
            domain=_enum_of(Domain, data.get("domain"), Domain.GENERAL),
            science_field=_enum_of(ScienceField, data.get("scienceField"), ScienceField.NONE),
            task_kind=_enum_of(TaskKind, data.get("taskKind"), TaskKind.NONE),
        )


def _enum_of(enum_cls, value, fallback):
    """Значение перечисления по ответу модели; незнакомое значение дает запасное."""
    try:
        return enum_cls(value)
    except ValueError:
        return fallback
