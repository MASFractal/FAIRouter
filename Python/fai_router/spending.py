"""Счет хода: во что обошлись ответ, распознавание, судья и повторы, и оценка расхода судьи для цены
выбора. Итог задачи складывается из всех обращений к моделям. Обращения идут из разных мест и
потоков, а клиент судьи общий у всех ходов, поэтому счет лежит в контекстной переменной: заменам
распознавателя и судьи у вызывающего ничего передавать не нужно. Устроен так же, как Bill в версии
на C#."""

from __future__ import annotations

import contextvars
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from fai_router.routed_element import RoutedElement

# Цена модели за миллион токенов: (вход, выход)
Price = tuple[float, float]

_current: contextvars.ContextVar["Bill | None"] = contextvars.ContextVar("fai_router_bill", default=None)

# Объем обращений к судье в токенах без текста задачи и ответа, (вход, выход). Вход посчитан по длине
# промптов и схем, выход по длине типичного вердикта на короткую задачу; величины не измерялись.
# Те же, что в Bill версии на C#
RECOGNITION_TOKENS = (3300, 200)
STYLE_TOKENS = (1700, 50)
REVIEW_TOKENS = (1800, 150)

# Сколько токенов ответа видит судья: ANSWER_CHARS судьи содержания при трех символах на токен
VISIBLE_ANSWER_TOKENS = 8000


@dataclass(frozen=True)
class RoundCost:
    """Во что обошелся ход по прайсу, в валюте поставщика: фактические токены, умноженные на цены
    моделей. Скидка за кэш входа не учитывается, поэтому счет поставщика бывает ниже. Попытка, на
    которую поставщик ответил ошибкой, не считается: о ее расходе он не сообщает.

    None означает, что цена хотя бы одной модели неизвестна: итог без нее выглядел бы полным, а им
    не был. estimated истинно, если поставщик или исполнитель не сообщил токены хотя бы одного
    обращения и они оценены по длине текста."""

    # Ответ, который получил вызывающий
    answer: float | None = None
    # Распознавание задания, замер стиля и суд содержания, первые попытки
    judge: float | None = None
    # Пустые ответы отказавших кандидатов и повторы обращений к судье
    retries: float | None = None
    total: float | None = None
    estimated: bool = False


@dataclass(frozen=True)
class _Call:
    price: Price | None
    prompt_tokens: float
    completion_tokens: float
    estimated: bool

    @property
    def amount(self) -> float | None:
        if self.price is None:
            return None
        return (self.price[0] * self.prompt_tokens + self.price[1] * self.completion_tokens) * 1e-6


class Bill:
    """Обращения одного хода по порядку. Последняя попытка исполнителя дала ответ, предыдущие отказали."""

    def __init__(self) -> None:
        self.attempts: list[_Call] = []
        # Обращения к судье: цена у них одна, и известна она только роутеру
        self.served: list[tuple[float, float, bool]] = []
        self.repeated: list[tuple[float, float, bool]] = []

    def cost(self, judge_price: Price | None) -> RoundCost:
        """Итог хода по цене модели судьи judge_price."""
        served = [_Call(judge_price, *call) for call in self.served]
        repeated = [_Call(judge_price, *call) for call in self.repeated]
        answer = self.attempts[-1:]
        retries = [*self.attempts[:-1], *repeated]
        everything = [*answer, *served, *retries]
        return RoundCost(_sum(answer), _sum(served), _sum(retries), _sum(everything),
                         any(call.estimated for call in everything))


@contextmanager
def open_bill() -> Iterator[Bill]:
    """Счет на время хода: обращения внутри блока записываются в него."""
    bill = Bill()
    token = _current.set(bill)
    try:
        yield bill
    finally:
        _current.reset(token)


def note_attempt(candidate: "RoutedElement", prompt_tokens: int, completion_tokens: int,
                 input_tokens: float, text: str) -> None:
    """Попытка исполнителя. Исполнитель, вернувший просто строку, токенов не сообщает: они оцениваются
    по объему диалога и длине ответа. Поправка цены кандидата сюда не входит: она оценивает рассуждения
    заранее, а токены рассуждений уже есть в ответе поставщика."""
    bill = _current.get()
    if bill is None:
        return
    price = (candidate.dpmt_inp, candidate.dpmt_outp) if candidate.has_known_price else None
    reported = prompt_tokens > 0 or completion_tokens > 0
    bill.attempts.append(_Call(price, prompt_tokens if reported else input_tokens,
                               completion_tokens if reported else _tokens(text), not reported))


def note_service(usage: tuple[int, int], messages: list[dict[str, Any]], schema: dict[str, Any], text: str,
                 repeat: bool) -> None:
    """Обращение к судье. Токены не сообщены, тогда они оцениваются по длине сообщений, схемы и ответа."""
    bill = _current.get()
    if bill is None:
        return
    reported = any(usage)
    prompt = usage[0] if reported else _tokens(json.dumps([messages, schema], ensure_ascii=False))
    (bill.repeated if repeat else bill.served).append((prompt, usage[1] if reported else _tokens(text), not reported))


def expected_judge_cost(task_tokens: float, answer_tokens: float, price: Price | None, measured: bool) -> float:
    """Ожидаемый расход судьи на ход, для цены выбора. У всех кандидатов он один, поэтому порядок цен
    не меняет, а меняет их отношение: на коротком запросе модель в двадцать раз дешевле дает те же
    несколько процентов итога, и платить за это качеством незачем. Распознавание к моменту выбора уже
    оплачено, но в итог задачи входит, а цена выбора оценивает именно его. Цена судьи неизвестна,
    тогда ноль. measured ложно, если ответ не оценивается и судья зовется только для распознавания."""
    if price is None:
        return 0.0
    calls = [(RECOGNITION_TOKENS[0] + task_tokens, RECOGNITION_TOKENS[1])]
    if measured:
        visible = min(answer_tokens, VISIBLE_ANSWER_TOKENS)
        calls += [(STYLE_TOKENS[0] + visible, STYLE_TOKENS[1]),
                  (REVIEW_TOKENS[0] + task_tokens + visible, REVIEW_TOKENS[1])]
    return sum(_Call(price, prompt, completion, True).amount for prompt, completion in calls)


def _sum(calls: list[_Call]) -> float | None:
    amounts = [call.amount for call in calls]
    return None if None in amounts else sum(amounts, 0.0)


def _tokens(text: str) -> float:
    # Импорт здесь: сервисы признаков сами обращаются к модели через этот модуль
    from fai_router.services import InputFeaturesService

    return len(text) / InputFeaturesService.EST_SYMBOL_PER_TOKEN
