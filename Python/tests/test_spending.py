import json

import numpy as np
import pytest

from fai_router import Completion, FaiRouter, RouteWeights, Settings, spending
from fai_router.llm.json_call import MAX_TOKENS
from fai_router.routed_element import RoutedElement

JUDGE_PRICE = (10.0, 100.0)


class MeteredLlm:
    """Судья, который, как поставщик, сообщает расход каждого обращения: 100 токенов входа и 10
    выхода. Обрезает первый ответ схемы из cut, заказанный объем задает symbols."""

    def __init__(self, cut=(), symbols=1500):
        self.cut = list(cut)
        self.symbols = symbols

    def complete_full(self, messages, schema=None, schema_name="answer", **kwargs):
        assert kwargs["max_tokens"] == MAX_TOKENS
        finish = "length" if schema_name in self.cut else "stop"
        if schema_name in self.cut:
            self.cut.remove(schema_name)
        reply = json.dumps(_verdict(schema_name, self.symbols))
        return {"choices": [{"message": {"content": reply}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def _verdict(schema_name, symbols=1500):
    if schema_name == "input_specifications":
        return {"styleType": "Scientific", "symbolLength": symbols, "language": "ru"}
    if schema_name == "content_review":
        return {"claims": [], "points": [], "constraints": [], "expertLevel": 0.7, "completeness": 0.9,
                "instructionFollowing": 1.0, "reasoning": 0.9, "expertise": 0.8, "structureContent": 0.8,
                "sourceQuality": 0.7, "fitForPurpose": 0.8, "issues": []}
    return {"styleType": "Scientific", "termDensity": 0.65, "formalityScore": 0.85}


def candidates():
    shared = np.ones(Settings.full_dim())
    return [RoutedElement("leader", tps=200, dpmt_inp=1.0, dpmt_outp=2.0, ideal_match_vector=shared * 2),
            RoutedElement("spare", tps=200, dpmt_inp=3.0, dpmt_outp=4.0, ideal_match_vector=shared.copy())]


def make_router(execute, llm=None, judge_price=JUDGE_PRICE, elements=None):
    return FaiRouter(elements or candidates(), execute, llm=llm or MeteredLlm(), judge_price=judge_price,
                     weights=RouteWeights(1, 0, 0, 0.0))


# Одно обращение к судье: 100 токенов входа и 10 выхода по его цене
JUDGE_CALL = (10 * 100 + 100 * 10) * 1e-6


def test_cost_separates_answer_from_judge():
    """Итог задачи это ответ плюс три обращения к судье: задание, стиль и содержание. Обращения из
    потоков замера попадают в счет своего хода."""
    answer = make_router(lambda candidate, messages: Completion("ответ", 1000, 500)).ask("Напиши обзор")

    assert answer.winner == "leader"
    assert answer.cost.answer == pytest.approx((1 * 1000 + 2 * 500) * 1e-6)
    assert answer.cost.judge == pytest.approx(3 * JUDGE_CALL)
    assert answer.cost.retries == 0
    assert answer.cost.total == pytest.approx(answer.cost.answer + answer.cost.judge)
    assert not answer.cost.estimated


def test_empty_answer_is_paid_and_failure_is_not():
    """Пустой ответ поставщик считает, поэтому он идет в повторы. Об ошибке расхода никто не сообщил,
    и она в счет не попадает."""
    def empty_then_spare(candidate, messages):
        return Completion("" if candidate.name == "leader" else "ответ", 1000, 500)

    def broken_then_spare(candidate, messages):
        if candidate.name == "leader":
            raise OSError("обрыв связи")
        return Completion("ответ", 1000, 500)

    empty = make_router(empty_then_spare).ask("Напиши обзор").cost
    assert empty.answer == pytest.approx((3 * 1000 + 4 * 500) * 1e-6)
    assert empty.retries == pytest.approx((1 * 1000 + 2 * 500) * 1e-6)
    assert empty.total == pytest.approx(empty.answer + empty.judge + empty.retries)

    broken = make_router(broken_then_spare).ask("Напиши обзор").cost
    assert broken.answer == empty.answer and broken.retries == 0


def test_repeat_of_judge_goes_to_retries():
    """Ответ судьи обрезан по потолку: повтор виден отдельной строкой счета."""
    llm = MeteredLlm(cut=["content_review"])
    cost = make_router(lambda candidate, messages: Completion("ответ", 1000, 500), llm).ask("Напиши обзор").cost

    assert cost.judge == pytest.approx(3 * JUDGE_CALL)
    assert cost.retries == pytest.approx(JUDGE_CALL)


def test_unknown_price_leaves_total_unknown():
    """Без цены судьи ответ по-прежнему посчитан, а итог нет: он выглядел бы полным, а им не был."""
    cost = make_router(lambda candidate, messages: Completion("ответ", 1000, 500), judge_price=None).ask("Напиши обзор").cost
    assert cost.answer == pytest.approx((1 * 1000 + 2 * 500) * 1e-6)
    assert cost.judge is None and cost.total is None

    unpriced = RoutedElement("leader", tps=200, dpmt_inp=-1.0, dpmt_outp=-1.0, ideal_match_vector=np.ones(Settings.full_dim()))
    cost = make_router(lambda candidate, messages: Completion("ответ", 1000, 500), elements=[unpriced]).ask("Напиши обзор").cost
    assert cost.answer is None and cost.total is None and cost.judge is not None


def test_tokens_are_estimated_when_nobody_reports_them():
    """Исполнитель вернул строку: токены оценены по длине текста, и счет это отмечает."""
    prompt = "x" * 300
    answer = make_router(lambda candidate, messages: "y" * 600).ask(prompt)

    assert answer.prompt_tokens == 0
    assert answer.cost.answer == pytest.approx((1 * 100 + 2 * 200) * 1e-6)
    assert answer.cost.estimated


def test_expected_judge_cost_follows_task_and_answer():
    """Распознавание платится всегда, стиль и суд только с замером; судья видит ответ не целиком."""
    price = (1.0, 0.0)
    recognition = spending.RECOGNITION_TOKENS[0] + 10
    measured = recognition + spending.STYLE_TOKENS[0] + 100 + spending.REVIEW_TOKENS[0] + 10 + 100

    assert spending.expected_judge_cost(10, 100, None, True) == 0
    assert spending.expected_judge_cost(10, 100, price, False) == pytest.approx(recognition * 1e-6)
    assert spending.expected_judge_cost(10, 100, price, True) == pytest.approx(measured * 1e-6)
    assert spending.expected_judge_cost(10, 10 ** 7, price, True) \
        == spending.expected_judge_cost(10, spending.VISIBLE_ANSWER_TOKENS, price, True)


@pytest.mark.parametrize("symbols, judge_price, winner", [
    (30, None, "cheap"),           # без цены судьи выбор как прежде, по одной цене ответа
    (30, (0.15, 0.6), "strong"),   # короткий запрос: разница ответов тонет в расходе судьи
    (30000, (0.15, 0.6), "cheap"), # длинный ответ: экономия настоящая
])
def test_judge_cost_flattens_price_gap_only_on_short_tasks(symbols, judge_price, winner):
    """Экономный профиль больше не отдает качество за экономию, которой в итоге задачи нет."""
    shared = np.ones(Settings.full_dim())
    elements = [RoutedElement("cheap", tps=100, dpmt_inp=0.04, dpmt_outp=0.08, ideal_match_vector=shared.copy()),
                RoutedElement("strong", tps=100, dpmt_inp=2.0, dpmt_outp=10.0, ideal_match_vector=shared * 2)]
    router = FaiRouter(elements, lambda candidate, messages: Completion("ответ", 10, 10), llm=MeteredLlm(symbols=symbols),
                       judge_price=judge_price, weights=RouteWeights(0.3, 0.60, 0.10, 0.0), measure=False)

    assert router.ask("Какая столица у Австралии?").winner == winner


def test_factory_takes_judge_price_from_prices_and_catalog():
    prices = {"my/own-model": (1.0, 2.0), "openai/gpt-4o-mini": (0.15, 0.6)}
    priced = FaiRouter.from_openai_compatible("https://example.test/v1", "k", ["my/own-model"], prices=prices, measure=False)
    assert priced.judge_price == (0.15, 0.6)
    assert [candidate.name for candidate in priced.candidates] == ["my/own-model"]

    def catalog():
        from fai_router.catalog import ModelInfo
        from fai_router.enums import Capability

        return [ModelInfo("a/model", "a", 1, 2, 0, 0, Capability.ALL), ModelInfo("openai/gpt-4o-mini", "j", 5, 6, 0, 0, Capability.ALL)]

    listed = FaiRouter.from_openai_compatible("https://example.test/v1", "k", ["a/model"], catalog_fetch=catalog, measure=False)
    assert listed.judge_price == (5, 6)

    unknown = FaiRouter.from_openai_compatible("https://example.test/v1", "k", ["my/own-model"],
                                               prices={"my/own-model": (1.0, 2.0)}, measure=False)
    assert unknown.judge_price is None
