"""Судья содержания, разбор всех расхождений с заданием и признаки задачи.

Контрольная пара повторяет главный пример владельца: структура выполнена безупречно, а содержание
провалено. Сверка формы ставит отлично, итоговая оценка обязана провалить, а критик обязан назвать
каждое расхождение с заданием. Числа общие с C#-тестом (Mas.Core.Tests/RouterContentTests.cs)."""

import json

import numpy as np
import pytest

from fai_router import content_review as cr
from fai_router.enums import Style, TaskKind
from fai_router.judge import Judge
from fai_router.llm.content_judge import from_json, user_message
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import InputFeatures


def report_order() -> Specifications:
    return Specifications(style_type=Style.OFFICIAL_BUSINESS, symbol_length=3000, word_length=450,
                          section_count=3, table_count=1, has_references=True, language="ru",
                          task_kind=TaskKind.ANALYTICAL_REPORT, expert_level=0.8,
                          required_points=["сравнить три тарифа по цене", "вывод о рентабельности"],
                          constraints=["цены в рублях"])


# Ответ модели-судьи на отчет, где таблица есть, но цифры выдуманы, а вывода нет
EMPTY_REPORT = json.dumps({
    "claims": [{"text": "Тариф Базовый стоит 990 рублей", "truth": 0.2},
               {"text": "Рентабельность выросла на 40%", "truth": 0.1}],
    "points": [{"point": "сравнить три тарифа по цене", "coverage": 0.5},
               {"point": "вывод о рентабельности", "coverage": 0.0}],
    "constraints": [{"constraint": "цены в рублях", "met": True}],
    "expertLevel": 0.3,
    "completeness": 0.3, "instructionFollowing": 1.0, "reasoning": 0.2, "expertise": 0.3,
    "structureContent": 0.1, "sourceQuality": 0.0, "fitForPurpose": 0.1,
    "issues": ["в таблице выдуманные цены", "вывода о рентабельности нет"],
}, ensure_ascii=False)


def test_form_perfect_content_failed_gives_failing_assessment():
    order = report_order()
    content = from_json(order, EMPTY_REPORT)
    critic = Judge.criticize(order, order, content)

    assert 1 - critic.form_deviation == pytest.approx(1.0)
    assert content.get(cr.FACTUALITY) == pytest.approx(0.15)
    # Полнота по поштучным пунктам (0.5 и 0), а не по общей оценке судьи (0.3)
    assert content.get(cr.COMPLETENESS) == pytest.approx(0.25)
    assert content.get(cr.INSTRUCTION_FOLLOWING) == pytest.approx(1.0)
    # Среднее по восьми критериям: 0.15, 0.25, 1.0, 0.2, 0.3, 0.1, 0.0, 0.1
    assert content.score == pytest.approx(2.1 / 8)
    assert Judge.assess(critic, content) == pytest.approx(0.7 * 2.1 / 8 + 0.3)
    assert Judge.assess(critic, content) < 0.6


def test_critic_names_every_deviation_from_the_order():
    order = report_order()
    critic = Judge.criticize(order, order, from_json(order, EMPTY_REPORT))

    # 20 пунктов формы и 10 содержания: 2 смысловых пункта, 1 ограничение, экспертность,
    # 2 утверждения, расчеты, наполнение, источники, пригодность
    assert len(critic.deviations) == 30
    assert len([item for item in critic.deviations if item.content]) == 10
    assert len(critic.mismatches) == 9
    assert critic.total_deviation == pytest.approx(7.425 / 30)
    assert critic.mismatches[0].field == "Смысловой пункт «вывод о рентабельности»"
    assert critic.mismatches[0].actual == "не раскрыт"
    fields = {item.field: item for item in critic.deviations}
    assert fields["Смысловой пункт «сравнить три тарифа по цене»"].actual == "раскрыт частично"
    assert fields["Ограничение «цены в рублях»"].deviation == 0
    assert fields["Экспертность"].deviation == pytest.approx(0.5 / 0.8)
    assert fields["Факт «Рентабельность выросла на 40%»"].deviation == pytest.approx(0.9)
    report = Judge.report(critic, from_json(order, EMPTY_REPORT))
    assert "Смысловой пункт «вывод о рентабельности»: заказано раскрыть, получено не раскрыт" in report
    assert "- в таблице выдуманные цены" in report


def test_violated_constraint_is_its_own_line():
    order = report_order()
    verdict = json.loads(EMPTY_REPORT)
    verdict["constraints"] = [{"constraint": "цены в рублях", "met": False}]
    content = from_json(order, json.dumps(verdict, ensure_ascii=False))
    critic = Judge.criticize(order, order, content)

    assert content.get(cr.INSTRUCTION_FOLLOWING) == pytest.approx(0.0)
    violated = next(item for item in critic.mismatches if item.field == "Ограничение «цены в рублях»")
    assert (violated.actual, violated.deviation) == ("нарушено", 1.0)


def test_criteria_that_do_not_apply_leave_the_average():
    order = Specifications(task_kind=TaskKind.STORY)
    content = from_json(order, json.dumps({"completeness": 0.8, "instructionFollowing": 0.0, "reasoning": 0.8,
                                           "expertise": 0.8, "structureContent": 0.8, "sourceQuality": 0.0,
                                           "fitForPurpose": 0.8}))

    assert content.get(cr.FACTUALITY) is None
    assert content.get(cr.INSTRUCTION_FOLLOWING) is None
    assert content.get(cr.SOURCE_QUALITY) is None
    assert content.expert_level is None
    assert content.score == pytest.approx(0.8)


def test_without_content_critic_is_form_only():
    order = report_order()
    critic = Judge.criticize(order, Specifications())
    assert len(critic.deviations) == 20
    assert critic.total_deviation == pytest.approx(critic.form_deviation)
    assert Judge.assess(critic, None) == pytest.approx(1 - critic.form_deviation)


def test_user_message_carries_numbered_points_and_constraints():
    text = user_message("Сравни тарифы", report_order(), "ответ")
    assert "1. сравнить три тарифа по цене" in text
    assert "2. вывод о рентабельности" in text
    assert "1. цены в рублях" in text
    assert "ЭКСПЕРТНОСТЬ ЗАДАНИЯ: 0.80" in text
    assert "НУЖНЫ ИСТОЧНИКИ: да" in text


def test_task_traits_enter_task_vector_but_not_answer_vector():
    plain = Specifications(style_type=Style.TECHNICAL, symbol_length=2000)
    demanding = Specifications(style_type=Style.TECHNICAL, symbol_length=2000, expert_level=0.9,
                               difficulty=0.7, factuality_demand=0.8, constraints=["a", "b", "c"])

    # Вектор ответа одинаков: судья формы требований заказа не видит
    assert np.allclose(plain.feature_vector(), demanding.feature_vector())

    single = InputFeatures(500, 1500, demanding).feature_vector()
    dialog = InputFeatures(500, 1500, demanding, turn_count=4).feature_vector()
    assert len(single) == Settings.full_dim()
    assert single[4] > 0 and single[5] > 0 and single[6] > 0 and single[3] > 0
    assert single[2] == 0 and dialog[2] > 0


def test_lists_survive_roundtrip():
    order = report_order()
    restored = Specifications.from_dict(order.to_dict())
    assert restored.required_points == order.required_points
    assert restored.constraints == order.constraints
    assert restored.task_kind == TaskKind.ANALYTICAL_REPORT
