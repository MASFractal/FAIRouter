from __future__ import annotations

import json
import math
from dataclasses import dataclass

from fai_router.enums import Domain, ScienceField, Style, TaskKind
from fai_router.llm import field_descriptions, json_call, prompt_data
from fai_router.settings import Settings
from fai_router.specifications import enum_value

# Сколько знаков текста видит оценщик: тот же предел, что у судьи содержания. Прежде текст уходил
# целиком (бывало 62 925 знаков)
TEXT_CHARS = 24_000


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
    считается по разметке.

    Отдельное от судьи содержания обращение оставлено намеренно: стиль и предмет ответа модель
    оценивает по одному ответу, не видя задания. Судья видит задание, и в общем вызове предмет ответа
    списывался бы с предмета задания, а сверка «ответ о том же, о чем задание» потеряла бы смысл."""

    SYSTEM_PROMPT = (
        "Ты оцениваешь стиль, лексику и предмет присланного текста. Определи стиль, долю терминологии, "
        "формальность тона, предметную область, область науки и тип результата (что это за текст), верни результат "
        "строго в виде JSON по заданной схеме, без пояснений."
    )

    # Бюджет времени на оценку, все попытки вместе
    BUDGET = 120.0

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

    def __init__(self, llm=None, budget: float = BUDGET):
        # Свой клиент нужен, когда в одном процессе судят несколько моделей
        self._llm = llm
        self.budget = budget

    def assess(self, text: str) -> StyleAssessment:
        if not text or not text.strip():
            raise ValueError("Текст для оценки не может быть пустым.")
        tag = prompt_data.new_tag()
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT + " " + prompt_data.rule(tag)},
                    {"role": "user", "content": prompt_data.wrap(tag, "текст", prompt_data.clip(text, TEXT_CHARS))}]
        return json_call.ask(self._llm or Settings.require_llm(), messages, "style_assessment",
                             self.SCHEMA, read, self.budget)


def read(raw: str) -> StyleAssessment | None:
    """Разбор ответа: значение вне перечисления (и число вместо имени) негодно, оно вышло бы за
    границы кода «один из многих»; пропущенное поле остается по умолчанию."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        return None
    defaults = StyleAssessment()
    fields = {}
    for key, name, enum_cls in (("styleType", "style_type", Style), ("domain", "domain", Domain),
                                ("scienceField", "science_field", ScienceField), ("taskKind", "task_kind", TaskKind)):
        value = enum_value(enum_cls, data[key]) if key in data else getattr(defaults, name)
        if value is None:
            return None
        fields[name] = value
    for key, name in (("termDensity", "term_density"), ("formalityScore", "formality_score")):
        value = data.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        fields[name] = float(value)
    return StyleAssessment(**fields)
