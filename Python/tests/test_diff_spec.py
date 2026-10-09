import pytest

from fai_router.enums import ProgrammingLanguage, Style
from fai_router.judge import Judge
from fai_router.specifications import Specifications


def test_critic_sees_what_cosine_hides():
    """Критик находит провалы там, где косинус говорил «почти идеально». Объем идет одной строкой
    (символы), шкалы с границами сверяются разностью в долях размаха."""
    requested = Specifications(style_type=Style.SCIENTIFIC, symbol_length=3000, word_length=450,
                               paragraph_count=6, section_count=3, table_count=1, heading_depth=2,
                               avg_sentence_length=18, readability_score=30, term_density=0.6,
                               formality_score=0.9, language="ru", has_references=True)
    actual = Specifications(style_type=Style.CHILDREN, symbol_length=1500, word_length=240,
                            paragraph_count=6, section_count=3, table_count=0, heading_depth=2,
                            avg_sentence_length=6, readability_score=95, term_density=0.05,
                            formality_score=0.1, language="ru", has_references=False)
    diff = Judge.criticize(requested, actual)

    # 19 пунктов: объем одной строкой, четыре пункта предмета задачи совпадают (оба «не задано»)
    assert len(diff.deviations) == 19
    assert len(diff.mismatches) == 8
    # Стиль 1, объем 0,5, таблицы 1, длина предложения 12/18, читаемость 0,65, термины 0,55,
    # формальность 0,8, источники 1
    assert diff.total_deviation == pytest.approx((1 + 0.5 + 1 + 12 / 18 + 0.65 + 0.55 + 0.8 + 1) / 19)
    assert diff.mismatches[0].field == "Стиль"
    assert "Таблицы: заказано 1, получено 0" in str(diff)


def test_zero_requested_is_full_mismatch_only_when_actual_differs():
    same = Specifications(table_count=0)
    other = Specifications(table_count=2)
    assert next(d for d in Judge.criticize(same, same).deviations if d.field == "Таблицы").deviation == 0
    assert next(d for d in Judge.criticize(same, other).deviations if d.field == "Таблицы").deviation == 1


def deviation(critic, field):
    return next(item.deviation for item in critic.deviations if item.field == field)


def test_bounded_scales_use_absolute_difference():
    """Тот же пример, что в C# (MeasureTests): ответ формальнее заказа штрафуется не сильнее, чем
    такой же недобор."""
    over = deviation(Judge.criticize(Specifications(formality_score=0.2), Specifications(formality_score=0.9)),
                     "Формальность")
    under = deviation(Judge.criticize(Specifications(formality_score=0.9), Specifications(formality_score=0.2)),
                      "Формальность")
    assert over == pytest.approx(0.7, abs=1e-9)
    assert under == pytest.approx(over, abs=1e-9)


def test_only_stated_fields_are_checked_strictly():
    """Угаданные поля не сверяются, объем одной строкой, языки через нормализацию, незаказанные
    источники не штрафуются."""
    requested = Specifications(symbol_length=3000, word_length=450, table_count=0, language="RU",
                               explicit_fields=["symbolLength", "wordLength", "language"])
    actual = Specifications(symbol_length=3000, word_length=900, table_count=2, language="ru", has_references=True)
    critic = Judge.criticize(requested, actual)
    fields = [item.field for item in critic.deviations]

    assert "Объем в символах" in fields
    assert "Объем в словах" not in fields
    assert "Таблицы" not in fields
    assert "Ссылки на источники" not in fields
    assert deviation(critic, "Язык") == 0
    assert critic.form_deviation == pytest.approx(0.0, abs=1e-9)


def test_legacy_order_checks_every_field_without_punishing_sources():
    critic = Judge.criticize(Specifications(language="ru-RU"), Specifications(language="ru", has_references=True))
    assert deviation(critic, "Ссылки на источники") == 0
    assert deviation(critic, "Язык") == 0
    assert str(next(item for item in critic.deviations if item.field == "Таблицы")) == "Таблицы: заказано 0, получено 0"


def test_unknown_language_and_unlabeled_code_are_not_checked():
    requested = Specifications(language="ru", programming_language=ProgrammingLanguage.PYTHON, code_block_count=1,
                               explicit_fields=["programmingLanguage", "language"])
    actual = Specifications(language=None, code_block_count=1)
    fields = [item.field for item in Judge.criticize(requested, actual).deviations]
    assert "Язык" not in fields and "Язык программирования" not in fields


def test_empty_critic_averages_to_zero():
    from fai_router.diff_spec import DiffSpec

    assert DiffSpec([]).total_deviation == 0 and DiffSpec([]).form_deviation == 0
    # Без явных полей формы сверяется только предмет задачи
    assert len(Judge.criticize(Specifications(explicit_fields=[]), Specifications()).deviations) == 3
