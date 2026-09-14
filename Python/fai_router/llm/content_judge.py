"""Судья содержания: одно обращение к модели по строгой схеме.

Оценивает то, что сверка формы не видит: верность фактов, полноту по сути, выполнение указаний,
рассуждения, глубину, наполненность структуры, источники и пригодность для дела. По каждому
смысловому пункту и каждому ограничению заказа судья отвечает отдельно, а уровень экспертности
ответа называет по шкале экспертности заказа: критик сверяет с заданием каждую из этих величин.
Факты проверяются так же, как в рейтинге фактологии арены: из ответа выписываются атомарные
проверяемые утверждения, у каждого своя вероятность истинности. Без проверки по вебу эту
вероятность ставит сама модель-судья; хост с веб-поиском передает проверку функцией verify."""

from __future__ import annotations

import json
from typing import Any, Callable

from fai_router import content_review as cr
from fai_router.content_review import ConstraintCheck, ContentCriterion, ContentReview, FactClaim, PointCoverage
from fai_router.llm.client import OpenRouterClient
from fai_router.settings import Settings
from fai_router.specifications import Specifications

# Сколько утверждений проверяется: больше дорого, меньше не хватает для средней
MAX_CLAIMS = 12
_ANSWER_CHARS = 24_000

SYSTEM_PROMPT = (
    "Ты строгий эксперт-приемщик. Оцени СОДЕРЖАНИЕ ответа на задание, а не оформление: объем, "
    "число разделов и таблиц проверяет код. Выпиши до 12 атомарных проверяемых утверждений "
    "ответа (даты, числа, имена, нормы, характеристики) и для каждого вероятность, что оно "
    "верно; мнения, оценки и вымысел не выписывай. По каждому смысловому пункту задания, в том "
    "же порядке, оцени, насколько он раскрыт; по каждому ограничению, в том же порядке, "
    "соблюдено ли оно. Уровень экспертности ответа оцени по той же шкале, что и экспертность "
    "задания. Затем оцени критерии от 0 до 1 по опорным точкам. В issues перечисли конкретные "
    "замечания по содержанию: что именно неверно или упущено и где. Ответ хорош, значит issues пусто."
)

# Опорные точки; текст общий с версией на C#
POINTS = (
    "По каждому смысловому пункту задания в том же порядке: насколько он раскрыт, 0-1. 1 - "
    "раскрыт по сути; 0.5 - упомянут без раскрытия; 0 - отсутствует или раскрыт неверно. "
    "Пусто, если пунктов нет."
)
CONSTRAINTS = "По каждому ограничению задания в том же порядке: соблюдено ли оно. Пусто, если ограничений нет."
EXPERT_LEVEL = (
    "Уровень экспертности самого ответа, 0-1, по той же шкале, что экспертность задания: 0.1 - "
    "бытовой уровень; 0.4 - грамотный пользователь; 0.7 - специалист; 0.9 - эксперт."
)
COMPLETENESS = (
    "Раскрыты ли смысловые пункты задания по сути, 0-1. 1 - каждый пункт раскрыт содержательно; "
    "0.6 - часть пунктов упомянута без раскрытия; 0.3 - раскрыта меньшая часть; 0 - ответ не о том."
)
INSTRUCTION_FOLLOWING = "Доля выполненных явных ограничений задания, 0-1. Если ограничений нет, 1."
REASONING = (
    "Верность рассуждений и расчетов, 0-1. 1 - выводы следуют из данных, числа сходятся; 0.5 - "
    "есть недоказанные выводы или мелкие ошибки в расчетах; 0 - выводы противоречат данным или "
    "расчеты неверны. Если рассуждений и расчетов нет, оцени логику изложения."
)
EXPERTISE = (
    "Глубина, которой ждет специалист области, 0-1. 0.2 - общие слова, подошедшие бы к любой "
    "задаче; 0.5 - грамотно, но поверхностно; 0.8 - конкретика, термины и нюансы по делу; "
    "1 - уровень опытного профессионала."
)
STRUCTURE_CONTENT = (
    "Содержательность структуры, 0-1: таблицы, списки и разделы наполнены данными по делу. "
    "1 - каждая строка несет содержание; 0.5 - часть строк пустые, повторяются или общие; "
    "0 - структура есть, а содержания в ней нет или оно выдумано."
)
SOURCE_QUALITY = (
    "Качество источников, 0-1: источники правдоподобно существуют, относятся к делу и "
    "подтверждают утверждения. 1 - все такие; 0.5 - часть не по делу или непроверяема; 0 - "
    "источники выдуманы или их нет. Если источники не нужны, 1."
)
FIT_FOR_PURPOSE = (
    "Пригодность для дела, 0-1: можно ли отдать результат заказчику как есть. 1 - как есть; "
    "0.7 - после мелкой правки; 0.4 - нужна существенная переделка; 0 - непригоден."
)


def _criterion(description: str) -> dict[str, Any]:
    return {"type": "number", "minimum": 0, "maximum": 1, "description": description}


def _array(description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "description": description,
            "items": {"type": "object", "properties": properties, "required": list(properties),
                      "additionalProperties": False}}


SCHEMA = {
    "type": "object",
    "properties": {
        "claims": _array("До 12 атомарных проверяемых утверждений ответа", {
            "text": {"type": "string", "description": "Утверждение одной фразой"},
            "truth": {"type": "number", "minimum": 0, "maximum": 1, "description": "Вероятность, что утверждение верно"},
        }),
        "points": _array(POINTS, {
            "point": {"type": "string", "description": "Пункт задания"},
            "coverage": {"type": "number", "minimum": 0, "maximum": 1, "description": "Насколько раскрыт"},
        }),
        "constraints": _array(CONSTRAINTS, {
            "constraint": {"type": "string", "description": "Ограничение задания"},
            "met": {"type": "boolean", "description": "Соблюдено ли"},
        }),
        "expertLevel": _criterion(EXPERT_LEVEL),
        "completeness": _criterion(COMPLETENESS),
        "instructionFollowing": _criterion(INSTRUCTION_FOLLOWING),
        "reasoning": _criterion(REASONING),
        "expertise": _criterion(EXPERTISE),
        "structureContent": _criterion(STRUCTURE_CONTENT),
        "sourceQuality": _criterion(SOURCE_QUALITY),
        "fitForPurpose": _criterion(FIT_FOR_PURPOSE),
        "issues": {"type": "array", "items": {"type": "string"},
                   "description": "Конкретные замечания по содержанию"},
    },
    "required": ["claims", "points", "constraints", "expertLevel", "completeness", "instructionFollowing",
                 "reasoning", "expertise", "structureContent", "sourceQuality", "fitForPurpose", "issues"],
    "additionalProperties": False,
}


class ContentJudge:
    """Судья содержания. verify: проверка утверждения по внешнему источнику, вероятность или
    None, если проверить не удалось; не задана, тогда вероятность ставит модель-судья."""

    def __init__(self, llm: OpenRouterClient | None = None,
                 verify: Callable[[str], float | None] | None = None):
        self._llm = llm
        self._verify = verify

    def review(self, task: str, requested: Specifications, answer: str) -> ContentReview:
        if not answer or not answer.strip():
            raise ValueError("Ответ не может быть пустым.")
        client = self._llm or Settings.require_llm()
        raw = client.complete(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_message(task, requested, answer)}],
            schema=SCHEMA, schema_name="content_review",
        )
        verdict = json.loads(raw)
        claims = []
        for claim in claims_of(verdict):
            checked = self._verify(claim.text) if self._verify is not None else None
            claims.append(claim if checked is None else FactClaim(claim.text, min(max(float(checked), 0.0), 1.0)))
        return build(requested, verdict, claims)


def from_json(requested: Specifications, raw: str) -> ContentReview:
    """Оценка по готовому ответу модели-судьи, без проверки утверждений по вебу. Нужна хосту,
    который хранит вердикты, и тестам: они проверяют сборку оценки без обращения к модели."""
    verdict = json.loads(raw)
    return build(requested, verdict, claims_of(verdict))


def claims_of(verdict: dict[str, Any]) -> list[FactClaim]:
    """Утверждения из ответа модели: не больше MAX_CLAIMS, пустые пропускаются."""
    claims = []
    for item in [item for item in (verdict.get("claims") or []) if str(item.get("text") or "").strip()][:MAX_CLAIMS]:
        claims.append(FactClaim(str(item["text"]).strip(), _clamp(item.get("truth", 0.0))))
    return claims


def build(requested: Specifications, verdict: dict[str, Any], claims: list[FactClaim]) -> ContentReview:
    """Оценка по ответу модели. Пункты и ограничения берутся в порядке заказа: пропущенный судьей
    пункт получает общую полноту, пропущенное ограничение считается соблюденным, если общая оценка
    выполнения указаний не ниже половины. Критерий, который к задаче не относится, остается пустым."""

    def score(key: str) -> float:
        return _clamp(verdict.get(key, 1.0))

    points = verdict.get("points") or []
    constraints = verdict.get("constraints") or []
    coverage = [PointCoverage(point, _clamp(points[i].get("coverage", 0.0)) if i < len(points) else score("completeness"))
                for i, point in enumerate(requested.required_points)]
    checks = [ConstraintCheck(constraint, bool(constraints[i].get("met")) if i < len(constraints)
                              else score("instructionFollowing") >= 0.5)
              for i, constraint in enumerate(requested.constraints)]
    completeness = sum(item.coverage for item in coverage) / len(coverage) if coverage else score("completeness")
    instruction = sum(1 for item in checks if item.met) / len(checks) if checks else None
    level = verdict.get("expertLevel")

    return ContentReview(
        criteria=[
            ContentCriterion(cr.FACTUALITY, ContentReview.factuality_of(claims)),
            ContentCriterion(cr.COMPLETENESS, completeness),
            ContentCriterion(cr.INSTRUCTION_FOLLOWING, instruction),
            ContentCriterion(cr.REASONING, score("reasoning")),
            ContentCriterion(cr.EXPERTISE, score("expertise")),
            ContentCriterion(cr.STRUCTURE_CONTENT, score("structureContent")),
            ContentCriterion(cr.SOURCE_QUALITY, score("sourceQuality") if requested.has_references else None),
            ContentCriterion(cr.FIT_FOR_PURPOSE, score("fitForPurpose")),
        ],
        claims=claims,
        issues=[str(issue) for issue in (verdict.get("issues") or []) if str(issue).strip()],
        points=coverage,
        constraint_checks=checks,
        expert_level=None if level is None else _clamp(level),
    )


def user_message(task: str, requested: Specifications, answer: str) -> str:
    points = _numbered(requested.required_points) if requested.required_points else "не выделены"
    constraints = _numbered(requested.constraints) if requested.constraints else "нет"
    clipped = answer if len(answer) <= _ANSWER_CHARS else answer[:_ANSWER_CHARS] + "\n[…ответ обрезан для судьи]"
    return (f"ЗАДАНИЕ:\n{task}\n\nСМЫСЛОВЫЕ ПУНКТЫ: {points}\n\nОГРАНИЧЕНИЯ: {constraints}\n\n"
            f"ЭКСПЕРТНОСТЬ ЗАДАНИЯ: {requested.expert_level:.2f}\n\n"
            f"НУЖНЫ ИСТОЧНИКИ: {'да' if requested.has_references else 'нет'}\n\nОТВЕТ:\n{clipped}")


def _numbered(items: list[str]) -> str:
    return "\n" + "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def _clamp(value: Any) -> float:
    return min(max(float(value), 0.0), 1.0)
