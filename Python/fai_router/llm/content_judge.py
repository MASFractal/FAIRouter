"""Судья содержания: одно обращение к модели по строгой схеме (при негодном ответе еще одно).

Оценивает то, что сверка формы не видит: верность фактов, полноту по сути, выполнение указаний,
рассуждения, глубину, наполненность структуры, источники и пригодность для дела. По каждому
смысловому пункту и каждому ограничению заказа судья отвечает отдельно, с номером пункта, а уровень
экспертности ответа называет по шкале экспертности заказа: критик сверяет с заданием каждую из этих
величин. Факты проверяются так же, как в замере фактологии: из ответа выписываются атомарные
проверяемые утверждения, у каждого своя вероятность истинности. Без проверки по вебу эту
вероятность ставит сама модель-судья; хост с веб-поиском передает проверку функцией verify.

Неполный ответ судьи (нет оценки хотя бы одного критерия) это сбой судьи, а не единица: оценки нет,
и автоотзыв по ходу не пишется. Задание и ответ идут судье внутри меток со случайным именем
(prompt_data), указания внутри них судья не исполняет. Тексты общие с версией на C# (ContentJudge)."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from fai_router import content_review as cr
from fai_router.content_review import ConstraintCheck, ContentCriterion, ContentReview, FactClaim, PointCoverage
from fai_router.llm import json_call, prompt_data
from fai_router.settings import Settings
from fai_router.specifications import Specifications

# Сколько утверждений проверяется: больше дорого, меньше не хватает для средней
MAX_CLAIMS = 12

# Сколько знаков ответа судья видит всегда; длинный заказ поднимает предел до MAX_ANSWER_CHARS
ANSWER_CHARS = 24_000

# Верхний предел показанного судье ответа: больше удорожает суд, а судья хуже держит длинный текст
MAX_ANSWER_CHARS = 48_000

# Бюджет времени на суд, все попытки вместе
BUDGET = 180.0

SYSTEM_PROMPT = (
    "Ты строгий эксперт-приемщик. Оцени СОДЕРЖАНИЕ ответа на задание, а не оформление: объем, "
    f"число разделов и таблиц проверяет код. Выпиши до {MAX_CLAIMS} атомарных проверяемых утверждений "
    "ответа (даты, числа, имена, нормы, характеристики) и для каждого вероятность, что оно "
    "верно; мнения, оценки и вымысел не выписывай. По каждому смысловому пункту задания, с его "
    "номером, оцени, насколько он раскрыт; по каждому ограничению, с его номером, соблюдено ли "
    "оно. Уровень экспертности ответа оцени по той же шкале, что и экспертность задания. Затем "
    "оцени критерии от 0 до 1 по опорным точкам. Длина и многословие сами по себе не достоинство: "
    "повторы, вода и лишнее снижают пригодность для дела, короткий точный ответ не хуже длинного. "
    "Если ответ обрезан для проверки, не считай упущенным пункт или ограничение, которые могли "
    "оказаться в отрезанной части: оценивай показанное. В issues перечисли конкретные замечания по "
    "содержанию: что именно неверно или упущено и где. Ответ хорош, значит issues пусто."
)

# Опорные точки; текст общий с версией на C# (ContentCriteriaDescriptions)
POINTS = (
    "По каждому смысловому пункту задания с его номером: насколько он раскрыт, 0-1. 1 - "
    "раскрыт по сути; 0.5 - упомянут без раскрытия; 0 - отсутствует или раскрыт неверно. "
    "Пусто, если пунктов нет."
)
CONSTRAINTS = "По каждому ограничению задания с его номером: соблюдено ли оно. Пусто, если ограничений нет."
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
    "1 - уровень опытного профессионала. Объем глубиной не считается."
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

# Критерии, без которых вердикт негоден: пропуск это не единица и не ноль, а сбой судьи
REQUIRED_SCORES = ("completeness", "instructionFollowing", "reasoning", "expertise", "structureContent",
                   "sourceQuality", "fitForPurpose")


def _criterion(description: str) -> dict[str, Any]:
    return {"type": "number", "minimum": 0, "maximum": 1, "description": description}


def _array(description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "description": description,
            "items": {"type": "object", "properties": properties, "required": list(properties),
                      "additionalProperties": False}}


SCHEMA = {
    "type": "object",
    "properties": {
        "claims": _array(f"До {MAX_CLAIMS} атомарных проверяемых утверждений ответа", {
            "text": {"type": "string", "description": "Утверждение одной фразой"},
            "truth": {"type": "number", "minimum": 0, "maximum": 1, "description": "Вероятность, что утверждение верно"},
        }),
        "points": _array(POINTS, {
            "index": {"type": "integer", "description": "Номер пункта в задании"},
            "point": {"type": "string", "description": "Пункт задания"},
            "coverage": {"type": "number", "minimum": 0, "maximum": 1, "description": "Насколько раскрыт"},
        }),
        "constraints": _array(CONSTRAINTS, {
            "index": {"type": "integer", "description": "Номер ограничения в задании"},
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

    def __init__(self, llm=None, verify: Callable[[str], float | None] | None = None, budget: float = BUDGET):
        self._llm = llm
        self._verify = verify
        self.budget = budget

    @property
    def model(self) -> str | None:
        """Модель судьи, если клиент ее знает; None, если не знает."""
        return getattr(self._llm or Settings.llm, "model", None)

    def review(self, task: str, requested: Specifications, answer: str) -> ContentReview:
        """Оценка содержания ответа на задание. Судья дважды ответил неполно или не по схеме, тогда
        json_call.InvalidModelAnswer; не уложился в бюджет, тогда TimeoutError."""
        if not answer or not answer.strip():
            raise ValueError("Ответ не может быть пустым.")
        tag = prompt_data.new_tag()
        messages = [{"role": "system", "content": SYSTEM_PROMPT + " " + prompt_data.rule(tag)},
                    {"role": "user", "content": user_message(task, requested, answer, tag)}]
        verdict = json_call.ask(self._llm or Settings.require_llm(), messages, "content_review", SCHEMA,
                                read, self.budget)
        claims = claims_of(verdict)
        return build(requested, verdict, claims if self._verify is None else self._checked(claims))

    def _checked(self, claims: list[FactClaim]) -> list[FactClaim]:
        """Проверка утверждений функцией хоста разом. Сбой или нечисловой ответ на одном утверждении
        оставляет ему оценку судьи, а не теряет весь разбор."""
        if not claims:
            return claims
        with ThreadPoolExecutor(max_workers=min(len(claims), MAX_CLAIMS)) as pool:
            truths = list(pool.map(self._check, (claim.text for claim in claims)))
        return [claim if truth is None else FactClaim(claim.text, _clamp(truth)) for claim, truth in zip(claims, truths)]

    def _check(self, claim: str) -> float | None:
        try:
            truth = self._verify(claim)
        except Exception:  # noqa: BLE001 - сбой проверки одного утверждения оставляет оценку судьи
            return None
        return float(truth) if _finite(truth) else None


def from_json(requested: Specifications, raw: str) -> ContentReview:
    """Оценка по готовому ответу модели-судьи, без проверки утверждений по вебу. Нужна хосту,
    который хранит вердикты, и тестам. Ограда ```json и текст вокруг допускаются; неполный ответ
    (нет оценки хотя бы одного критерия) это json_call.InvalidModelAnswer."""
    verdict = json_call.parse_answer(raw, read)
    if verdict is None:
        raise json_call.InvalidModelAnswer("Ответ судьи неполон или не по схеме: оценки нет.")
    return build(requested, verdict, claims_of(verdict))


def read(raw: str) -> dict[str, Any] | None:
    """Вердикт без оценки хотя бы одного критерия или с нечисловой оценкой негоден."""
    verdict = json.loads(raw)
    if not isinstance(verdict, dict) or not all(_finite(verdict.get(key)) for key in REQUIRED_SCORES):
        return None
    return verdict


def claims_of(verdict: dict[str, Any]) -> list[FactClaim]:
    """Утверждения из ответа модели, не больше MAX_CLAIMS. Без текста или без вероятности (или с
    нечисловой) утверждение отбрасывается: ноль по умолчанию объявлял бы его ложным."""
    claims = []
    for item in _objects(verdict.get("claims")):
        text = item.get("text")
        if isinstance(text, str) and text.strip() and _finite(item.get("truth")):
            claims.append(FactClaim(text.strip(), _clamp(item["truth"])))
    return claims[:MAX_CLAIMS]


def build(requested: Specifications, verdict: dict[str, Any], claims: list[FactClaim]) -> ContentReview:
    """Оценка по ответу модели. Пункты и ограничения сопоставляются по номеру; номеров нет, а число
    совпало, тогда по порядку. Пункт без оценки получает общую полноту, ограничение без оценки
    считается соблюденным, если общая оценка выполнения указаний не ниже половины. Критерий, который к
    задаче не относится, остается пустым."""
    points = requested.required_points[:Specifications.MAX_ITEMS]
    constraints = requested.constraints[:Specifications.MAX_ITEMS]
    overall_completeness = _clamp(verdict["completeness"])
    overall_instruction = _clamp(verdict["instructionFollowing"])

    covered = _match(len(points), _objects(verdict.get("points")), "coverage", _finite)
    met = _match(len(constraints), _objects(verdict.get("constraints")), "met", lambda value: isinstance(value, bool))
    coverage = [PointCoverage(point, _clamp(overall_completeness if covered[i] is None else covered[i]))
                for i, point in enumerate(points)]
    checks = [ConstraintCheck(constraint, overall_instruction >= 0.5 if met[i] is None else met[i])
              for i, constraint in enumerate(constraints)]

    completeness = sum(item.coverage for item in coverage) / len(coverage) if coverage else overall_completeness
    instruction = sum(1 for item in checks if item.met) / len(checks) if checks else None
    level = verdict.get("expertLevel")

    return ContentReview(
        criteria=[
            ContentCriterion(cr.FACTUALITY, ContentReview.factuality_of(claims)),
            ContentCriterion(cr.COMPLETENESS, completeness),
            ContentCriterion(cr.INSTRUCTION_FOLLOWING, instruction),
            ContentCriterion(cr.REASONING, _clamp(verdict["reasoning"])),
            ContentCriterion(cr.EXPERTISE, _clamp(verdict["expertise"])),
            ContentCriterion(cr.STRUCTURE_CONTENT, _clamp(verdict["structureContent"])),
            ContentCriterion(cr.SOURCE_QUALITY, _clamp(verdict["sourceQuality"]) if requested.has_references else None),
            ContentCriterion(cr.FIT_FOR_PURPOSE, _clamp(verdict["fitForPurpose"])),
        ],
        claims=claims,
        issues=[issue for issue in (verdict.get("issues") or []) if isinstance(issue, str) and issue.strip()],
        points=coverage,
        constraint_checks=checks,
        expert_level=_clamp(level) if _finite(level) else None,
    )


def answer_limit(requested: Specifications) -> int:
    """Сколько знаков ответа видит судья: заказанный объем с запасом в четверть, но не меньше
    ANSWER_CHARS и не больше MAX_ANSWER_CHARS."""
    return int(min(max(min(requested.symbol_length * 1.25, MAX_ANSWER_CHARS), ANSWER_CHARS), MAX_ANSWER_CHARS))


def user_message(task: str, requested: Specifications, answer: str, tag: str | None = None) -> str:
    """Сообщение судье: задание, пункты, ограничения и ответ в метках tag (новая, если не задана)."""
    tag = tag or prompt_data.new_tag()
    points = _numbered(requested.required_points, "не выделены")
    constraints = _numbered(requested.constraints, "нет")
    return (f"{prompt_data.wrap(tag, 'задание', task)}\n\n"
            f"{prompt_data.wrap(tag, 'смысловые пункты', points)}\n\n"
            f"{prompt_data.wrap(tag, 'ограничения', constraints)}\n\n"
            f"ЭКСПЕРТНОСТЬ ЗАДАНИЯ: {requested.expert_level:.2f}\n\n"
            f"НУЖНЫ ИСТОЧНИКИ: {'да' if requested.has_references else 'нет'}\n\n"
            + prompt_data.wrap(tag, "ответ", prompt_data.clip(answer, answer_limit(requested))))


def _match(count: int, judged: list[dict[str, Any]], key: str, valid: Callable[[Any], bool]) -> list[Any]:
    """Оценки судьи по номерам пунктов заказа. Номера есть у всех, тогда по номерам; номеров нет, а
    число совпало, тогда по порядку (прежний вид ответа); иначе сопоставить нельзя, и пункт получает
    общую оценку."""
    matched: list[Any] = [None] * count
    numbered = bool(judged) and all(_is_index(item.get("index")) for item in judged)
    for i, item in enumerate(judged):
        slot = item["index"] - 1 if numbered else i if len(judged) == count else -1
        if 0 <= slot < count and matched[slot] is None and valid(item.get(key)):
            matched[slot] = item[key]
    return matched


def _numbered(items: list[str], empty: str) -> str:
    if not items:
        return empty
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items[:Specifications.MAX_ITEMS]))


def _objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clamp(value: Any) -> float:
    return min(max(float(value), 0.0), 1.0)
