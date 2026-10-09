"""Выбор кандидата: жребий, температура, метрика R, планка, отсев по объему и запасной кандидат.
Те же проверки, что в C#-тестах SelectionTests.cs и ExecutionTests.cs."""

import random

import numpy as np
import pytest

from fai_router import env
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import RouteWeights, Settings, SufficiencyBar
from fai_router.specifications import Specifications
from fai_router.tracking import Tracert
from fai_router.training import Calibration

QUALITY_ONLY = RouteWeights(1, 0, 0, 0)
PRICE_ONLY = RouteWeights(0, 1, 0, 0)


def features():
    return InputFeaturesService.get_features("Разбери задачу и предложи решение.")


def rated(task, name, quality, price, experience=1000.0, tps=100.0, variance=0.01):
    """Кандидат с заданным прогнозом: вектор сонаправлен задаче, и прогноз равен множителю."""
    candidate = RoutedElement(name, tps=tps, dpmt_inp=price, dpmt_outp=price * 5,
                              ideal_match_vector=task.feature_vector() * quality)
    candidate.experience, candidate.score_variance = experience, variance
    return candidate


def test_sample_uses_the_true_maximum():
    task = features()
    low, high = rated(task, "низкий", 0.1, 1), rated(task, "высокий", 0.9, 1)
    rng = random.Random(7)
    assert all(env.sample([(0.0, low), (50.0, high)], RouteWeights(1, 0, 0, 30), rng) == 1 for _ in range(20))


def test_sample_is_reproducible_with_a_seed():
    task = features()
    group = [(0.1, rated(task, "a", 0.5, 1, experience=0)), (0.0, rated(task, "b", 0.5, 1, experience=0)),
             (-0.1, rated(task, "c", 0.5, 1, experience=0))]
    one, two = random.Random(42), random.Random(42)
    first = [env.sample(group, None, one) for _ in range(30)]
    assert first == [env.sample(group, None, two) for _ in range(30)]
    assert len(set(first)) > 1, "у новичков жребий должен давать разных победителей"


def test_prior_experience_cools_the_group():
    task = features()
    cold = rated(task, "холодный", 0.5, 1, experience=0)
    known = rated(task, "с рейтингом", 0.5, 1, experience=0)
    known.prior_experience = Settings.PRIOR_EXPERIENCE
    assert env.temperature([known]) < env.temperature([cold])


def test_nan_forecast_gets_the_worst_quality_not_poisons_the_group():
    task = features()
    broken = rated(task, "сломанный", 0.9, 1)
    broken.ideal_match_vector[0] = float("nan")
    top = env.get_top_k(task, [broken, rated(task, "a", 0.3, 1), rated(task, "b", 0.6, 1)], weights=QUALITY_ONLY)
    assert all(np.isfinite(score) for score, _ in top)
    assert top[0][1].name == "b"


def test_non_finite_numbers_are_rejected_on_the_element():
    candidate = RoutedElement("a")
    with pytest.raises(ValueError):
        candidate.ideal_match_vector = np.array([float("nan")])
    with pytest.raises(ValueError):
        candidate.tps = float("inf")
    with pytest.raises(ValueError):
        candidate.cost_ratio = -1
    with pytest.raises(ValueError):
        candidate.dpmt_inp = float("nan")


def test_unknown_price_ranks_as_the_most_expensive():
    """Неизвестная цена (минус единица у openrouter/auto) это худшая цена группы, а не NaN или «бесплатно»."""
    task = features()
    unknown, paid = rated(task, "неизвестная", 0.5, -1), rated(task, "платная", 0.5, 10)
    assert not unknown.has_known_price
    top = env.get_top_k(task, [unknown, paid], weights=PRICE_ONLY)
    assert top[0][1].name == "платная"
    assert all(np.isfinite(score) for score, _ in top)


def test_ties_are_broken_by_quality_then_name():
    task = features()
    forward = [e.name for _, e in env.get_top_k(task, [rated(task, "b", 0.5, 1), rated(task, "a", 0.5, 1)])]
    backward = [e.name for _, e in env.get_top_k(task, [rated(task, "a", 0.5, 1), rated(task, "b", 0.5, 1)])]
    assert forward == backward == ["a", "b"]


def test_close_prices_are_not_blown_up_and_free_models_stay_finite():
    """Цены, различающиеся на проценты, не раздуваются до единичного разброса (нижняя граница
    разброса), а бесплатная модель не уходит в минус бесконечность (пол цены от самой низкой)."""
    task = features()
    cheaper, better = rated(task, "дешевле на 1 %", 0.5, 0.99), rated(task, "лучше", 0.6, 1.0)
    assert env.get_top_k(task, [cheaper, better])[0][1] is better
    free = rated(task, "бесплатная", 0.5, 0.0)
    assert all(np.isfinite(score) for score, _ in env.get_top_k(task, [free, better]))


def test_quality_only_profile_with_a_bar_orders_by_quality():
    """Профиль «только качество» с планкой: прежде вес качества среди прошедших обнулялся, и
    порядок задавал каталог."""
    task = features()
    everyone = SufficiencyBar(0.5, Calibration(0, 10), prior_rate=0.99)
    chosen = env.get_sufficient(task, [rated(task, "слабая", 0.4, 1), rated(task, "сильная", 0.9, 1)],
                                everyone, weights=QUALITY_ONLY)
    assert chosen.reached and chosen.top[0][1].name == "сильная"


def test_cold_candidates_differ_by_their_forecast():
    """Холодный старт под планкой: новичок получает свой откалиброванный прогноз, стянутый к доле
    лайков. Прежде все новички получали одну долю лайков, и ниже планки не проходил ни один."""
    task = features()
    bar = SufficiencyBar(0.55, Calibration(10, -5), prior_rate=0.5)
    strong = rated(task, "сильный", 0.95, 1, experience=0)
    weak = rated(task, "слабый", 0.2, 0.1, experience=0)
    assert bar.sufficiency(0, 0.95) > bar.sufficiency(0, 0.2)
    chosen = env.get_sufficient(task, [weak, strong], bar)
    assert chosen.reached and [item[1].name for item in chosen.top] == ["сильный"]


def test_shortfall_draws_lots_among_the_strongest():
    """При недоборе ход разыгрывается среди сильнейших: иначе первый в списке получал все ходы."""
    task = features()
    unreachable = SufficiencyBar(0.99, Calibration(10, -5), prior_rate=0.3)
    cold = [rated(task, f"новичок{i}", 0.5, 1, experience=0) for i in range(4)]
    rng = random.Random(3)
    winners = {env.choose_sufficient(task, cold, unreachable, rng=rng).winner.name for _ in range(60)}
    assert len(winners) > 1
    assert env.choose_sufficient(task, cold, unreachable, rng=rng).bar_reached is False


def test_context_shortfall_takes_the_largest_window():
    task = features()
    task.input_len = 1_000_000
    small, large = rated(task, "малое окно", 0.9, 1), rated(task, "большое окно", 0.1, 1)
    small.context_window, large.context_window = 8_000, 200_000
    trace = env.choose(task, [small, large])
    assert trace.winner.name == "большое окно" and trace.context_shortfall
    task.input_len = 100
    assert not env.choose(task, [small, large], weights=QUALITY_ONLY).context_shortfall


def test_window_is_checked_against_the_whole_dialog():
    task = features()
    candidate = rated(task, "a", 0.5, 1)
    candidate.context_window = 10_000
    task.input_len, task.len_answer = 9_000, 2_000
    assert candidate.supports(task.input_specifications)
    assert not candidate.supports(task)


def test_failure_and_empty_answer_fall_back_and_are_recorded():
    task = features()
    leader, spare = rated(task, "лидер", 0.9, 1), rated(task, "запасной", 0.5, 1)
    trace = Tracert(winner=leader, top_k_elements=[leader, spare], input_feature_vector=task.feature_vector())
    assert env.execute(trace, lambda candidate: "  " if candidate is leader else "ответ") == "ответ"
    assert trace.winner is spare and trace.failed == ["лидер"] and trace.is_exploration
    assert trace.forecast == pytest.approx(spare.get_quality_score(trace.input_feature_vector))

    def timeout(candidate):
        if candidate is leader:
            raise TimeoutError("таймаут клиента")
        return "ответ"

    other = Tracert(winner=leader, top_k_elements=[leader, spare], input_feature_vector=task.feature_vector())
    assert env.execute(other, timeout) == "ответ" and other.failed == ["лидер"]


def test_recognition_failure_does_not_drop_the_turn():
    task = features()

    class Failing:
        def get_specifications(self, text):
            raise ValueError("обрезанный JSON")

    trace = env.route("Напиши обзор", [rated(task, "a", 0.5, 1)], specs=Failing(), rng=random.Random(1))
    assert trace.requested_spec is None and trace.winner.name == "a"


def test_route_uses_its_own_recognizer_and_dialog_volume():
    task = features()
    calls = []

    class Fixed:
        def get_specifications(self, text):
            calls.append(text)
            return Specifications(symbol_length=3000)

    trace = env.route("Напиши обзор", [rated(task, "a", 0.5, 1), rated(task, "b", 0.6, 1)],
                      specs=Fixed(), input_tokens=5000, rng=random.Random(1))
    assert calls == ["Напиши обзор"] and trace.forecast is not None
    assert trace.requested_spec.symbol_length == 3000

    candidate = rated(task, "a", 0.5, 1)
    short = candidate.get_cost(task)
    task.input_len = 50_000
    assert candidate.get_cost(task) > short


def test_features_dim_below_seven_is_rejected_and_extra_dims_stay_zero():
    Settings.FEATURES_DIM = 6
    try:
        with pytest.raises(ValueError):
            Settings.full_dim()
        Settings.FEATURES_DIM = 9
        vector = features().feature_vector()
        assert len(vector) == Settings.full_dim() and vector[7] == vector[8] == 0
    finally:
        Settings.FEATURES_DIM = 7


def test_temperature_on_a_full_size_catalog():
    """Температура на каталоге реального размера: 54 популярные модели со снимка. Безрейтинговые и
    рейтинговые лежат на одной шкале, группа из рейтинговых холоднее группы новичков, а жребий с
    множителем фабрики не отдает ход дальше первой пятерки."""
    from fai_router import benchmarks, catalog

    snapshot = benchmarks.default_snapshot()
    models = catalog.popular_model_infos()
    candidates = catalog.create_candidates([model.id for model in models], {model.id: model for model in models},
                                           benchmarks=snapshot, task_mean=None)
    assert len(candidates) >= 40
    informed = [candidate for candidate in candidates if candidate.prior_experience > 0]
    assert informed and len(informed) < len(candidates)
    task = features()
    weights = RouteWeights(0.5, 0.25, 0.25, Settings.PRIOR_TEMPERATURE_SCALE)
    top = env.get_top_k(task, candidates, topk=5, weights=weights)
    assert len(top) == 5 and all(np.isfinite(score) for score, _ in top)
    t = env.temperature([candidate for _, candidate in top], weights)
    assert 0 < t < Settings.PRIOR_TEMPERATURE_SCALE * np.sqrt(Settings.UNKNOWN_VARIANCE)
    assert env.temperature(informed[:5], weights) < env.temperature(
        [RoutedElement(f"новичок{i}") for i in range(5)], weights)
    rng = random.Random(5)
    names = {candidate.name for _, candidate in top}
    assert all(env.choose(task, candidates, rng=rng, weights=weights).winner.name in names for _ in range(50))
