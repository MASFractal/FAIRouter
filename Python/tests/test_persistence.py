import numpy as np
import pytest

from fai_router.enums import FeedbackType, Style
from fai_router.judge import Judge
from fai_router.persistence import SqliteTraceStore, SqliteWeightsStore
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert


def test_weights_roundtrip_is_exact(tmp_path):
    store = SqliteWeightsStore(str(tmp_path / "w.db"))
    good = RoutedElement("Подходящая")
    bad = RoutedElement("Неподходящая")
    judge = Judge()
    judge.transformer_w[0, 1] = 0.42
    mean = np.linspace(0, 1, Settings.full_dim())

    store.save_elements([good, bad])
    store.save_judge(judge)
    store.save_task_mean(mean)

    copies = [RoutedElement("Подходящая"), RoutedElement("Неподходящая"), RoutedElement("Невиданная")]
    judge_copy = Judge()
    assert store.load_elements(copies) == 2
    assert store.load_judge(judge_copy)
    assert np.array_equal(copies[0].ideal_match_vector, good.ideal_match_vector)
    assert np.array_equal(judge_copy.transformer_w, judge.transformer_w)
    assert np.array_equal(store.load_task_mean(), mean)
    assert not SqliteWeightsStore(str(tmp_path / "empty.db")).load_judge(Judge())


def test_wrong_dimension_is_rejected(tmp_path):
    store = SqliteWeightsStore(str(tmp_path / "w.db"))
    store.save_elements([RoutedElement("Подходящая")])
    with pytest.raises(ValueError, match="размерности"):
        store.load_elements([RoutedElement("Подходящая", ideal_match_vector=np.zeros(5))])


def test_trace_journal_accumulates_and_restores(tmp_path):
    traces = SqliteTraceStore(str(tmp_path / "t.db"))
    writer, coder = RoutedElement("Писатель"), RoutedElement("Кодер")
    science = Specifications(style_type=Style.SCIENTIFIC, symbol_length=4000, term_density=0.7)
    features = InputFeaturesService.get_features("задача")
    features.input_specifications = science
    trace = Tracert(winner=writer, top_k_elements=[writer, coder],
                    input_feature_vector=features.feature_vector(), is_exploration=True)

    rated = traces.append(trace, science, science, "научный обзор")
    traces.append(trace, science, None, "без отзыва")
    assert traces.count() == (2, 0)

    traces.set_feedback(rated, Feedback(FeedbackType.HUMAN, 0.9))
    assert traces.count() == (2, 1)
    with pytest.raises(ValueError):
        traces.set_feedback(999, Feedback())

    sample = traces.read_rated([writer, coder])
    assert len(sample) == 1
    assert sample[0].trace.winner is writer and sample[0].trace.is_exploration
    assert sample[0].requested.style_type == Style.SCIENTIFIC and sample[0].actual is not None
    assert len(sample[0].trace.input_feature_vector) == Settings.full_dim()
    assert traces.read_rated([coder]) == []

    assert traces.load_statistics([writer, coder]) == 1
    assert writer.experience == 1 and writer.score_variance == 0.0
    assert np.allclose(traces.get_feature_mean(), features.feature_vector())


def test_statistics_count_only_human_feedback(tmp_path):
    """Опыт и разброс считаются только по человеческим отзывам. Автоотзыв ставится на каждый ход,
    и по нему опыт рос сам собой: температура падала, разведка гасла по мнению собственного судьи."""
    traces = SqliteTraceStore(str(tmp_path / "t.db"))
    writer, coder = RoutedElement("Писатель"), RoutedElement("Кодер")
    features = InputFeaturesService.get_features("задача")
    trace = Tracert(winner=writer, top_k_elements=[writer, coder],
                    input_feature_vector=features.feature_vector())

    for _ in range(3):
        traces.set_feedback(traces.append(trace), Feedback(FeedbackType.AUTO, 0.8))
    traces.set_feedback(traces.append(trace), Feedback(FeedbackType.HUMAN, 1.0))

    assert traces.load_statistics([writer, coder]) == 1
    assert writer.experience == 1
    # В обучающую выборку автоотзывы при этом входят: роутер по ним учится, просто слабее
    assert len(traces.read_rated([writer, coder])) == 4
