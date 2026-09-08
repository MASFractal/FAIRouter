from fai_router.enums import Style
from fai_router.judge import Judge
from fai_router.specifications import Specifications


def test_critic_sees_what_cosine_hides():
    """Тот же пример, что в C#: критик находит 9 провалов из 16, косинус говорил «почти идеально»."""
    requested = Specifications(style_type=Style.SCIENTIFIC, symbol_length=3000, word_length=450,
                               paragraph_count=6, section_count=3, table_count=1, heading_depth=2,
                               avg_sentence_length=18, readability_score=30, term_density=0.6,
                               formality_score=0.9, language="ru", has_references=True)
    actual = Specifications(style_type=Style.CHILDREN, symbol_length=1500, word_length=240,
                            paragraph_count=6, section_count=3, table_count=0, heading_depth=2,
                            avg_sentence_length=6, readability_score=95, term_density=0.05,
                            formality_score=0.1, language="ru", has_references=False)
    diff = Judge.criticize(requested, actual)

    assert len(diff.deviations) == 16
    assert len(diff.mismatches) == 9
    assert abs(diff.total_deviation - 0.465) < 1e-3
    assert diff.mismatches[0].field == "Стиль"
    assert "Таблицы: заказано 1, получено 0" in str(diff)


def test_zero_requested_is_full_mismatch_only_when_actual_differs():
    same = Specifications(table_count=0)
    other = Specifications(table_count=2)
    assert next(d for d in Judge.criticize(same, same).deviations if d.field == "Таблицы").deviation == 0
    assert next(d for d in Judge.criticize(same, other).deviations if d.field == "Таблицы").deviation == 1
