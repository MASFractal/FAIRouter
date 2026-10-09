import random

import numpy as np

from fai_router import env
from fai_router.enums import FeedbackType, Style
from fai_router.judge import Judge
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert
from fai_router.training import JudgeTrainer, RouterTrainer, from_measurements

import pytest


def spec(style, symbols, terms, formality, sections=3):
    return Specifications(style_type=style, symbol_length=symbols, word_length=symbols // 7,
                          section_count=sections, term_density=terms, formality_score=formality)


def test_judge_learns_to_repeat_human():
    """Судья ошибается: косинус 1,0, а человек недоволен. В C# оценка ушла к нулю за 25 шагов."""
    requested = spec(Style.SCIENTIFIC, 4000, 0.7, 0.9)
    lookalike = spec(Style.SCIENTIFIC, 3900, 0.69, 0.89)
    judge = Judge()
    trainer = JudgeTrainer(judge, learning_rate=1.0)

    before = judge.get_score(requested, lookalike)
    first = trainer.train(requested, lookalike, 0.0)
    for _ in range(199):
        last = trainer.train(requested, lookalike, 0.0)

    assert before > 0.99
    assert judge.get_score(requested, lookalike) < 0.1
    assert last < first

    satisfied = Judge()
    JudgeTrainer(satisfied, 1.0).train(requested, lookalike, 1.0)
    assert satisfied.get_score(requested, lookalike) > 0.99


def test_router_separates_winner_from_rival_by_margin():
    features = InputFeaturesService.get_features("Напиши научный обзор")
    features.input_specifications = spec(Style.SCIENTIFIC, 4000, 0.7, 0.9)
    shared = np.full(Settings.full_dim(), 1.0 / Settings.FEATURES_DIM)
    good = RoutedElement("Подходящая", ideal_match_vector=shared.copy())
    bad = RoutedElement("Неподходящая", ideal_match_vector=shared.copy())
    trace = Tracert(winner=good, top_k_elements=[good, bad], input_feature_vector=features.feature_vector())
    trainer = RouterTrainer(learning_rate=0.05)
    like = Feedback(FeedbackType.HUMAN, 1.0)

    def gap():
        vector = features.feature_vector()
        return float(good.ideal_match_vector @ vector - bad.ideal_match_vector @ vector)

    assert gap() == 0
    for _ in range(200):
        loss = trainer.train(trace, like)
    assert gap() >= RouterTrainer.MARGIN - 1e-6
    assert loss == 0
    assert env.get_top_k(features, [bad, good])[0][1] is good


def test_prior_from_measurements_picks_best_model_per_task_type():
    """Тот же замер, что в model-quality-grid.md: четыре совпадения из четырех до обучения."""
    kinds = [(Style.SCIENTIFIC, 1500, 0.70, 0.90), (Style.CHILDREN, 600, 0.05, 0.15),
             (Style.OFFICIAL_BUSINESS, 700, 0.35, 0.95), (Style.TECHNICAL, 1000, 0.45, 0.50)]
    measured = {"gemini": [0.468, 0.726, 0.741, 0.572],
                "gpt": [0.532, 0.675, 0.765, 0.607],
                "claude": [0.634, 0.634, 0.650, 0.723]}

    def task(kind):
        features = InputFeaturesService.get_features("задача")
        features.input_specifications = spec(*kind)
        return features

    vectors = [task(kind).feature_vector() for kind in kinds]
    Settings.task_mean = np.mean(vectors, axis=0)
    candidates = [RoutedElement(name, tps=100, dpmt_inp=1, dpmt_outp=5,
                                ideal_match_vector=from_measurements(list(zip(vectors, scores))))
                  for name, scores in measured.items()]

    hits = 0
    for k, kind in enumerate(kinds):
        chosen = env.get_top_k(task(kind), candidates)[0][1].name
        best = max(measured, key=lambda name: measured[name][k])
        hits += chosen == best
    assert hits == 4


def test_three_task_types_separate_with_centering():
    """Как в exploration.md: вычитание среднего и возврат длины. С прежним правилом обучения,
    двигавшим и соперников, здесь было 18-20 запусков из 20; с градиентом только в победителя (как в
    C#, с 09.10.2026) 13 из 20 на тех же 300 ходах: соперников, которых на ходе не пробовали, отзыв
    больше не двигает, и закрепление на первом одобренном случается чаще."""
    kinds = [(Style.SCIENTIFIC, 4000, 0.70, 0.90, "Ученый"), (Style.TECHNICAL, 1200, 0.45, 0.50, "Инженер"),
             (Style.CHILDREN, 600, 0.05, 0.15, "Педагог")]

    def task(kind):
        features = InputFeaturesService.get_features("задача")
        features.input_specifications = spec(*kind[:4])
        return features

    Settings.task_mean = np.mean([task(kind).feature_vector() for kind in kinds], axis=0)

    def run(seed):
        rng = np.random.default_rng(seed)
        dice = random.Random(seed)
        candidates = [RoutedElement(kind[4], tps=100, dpmt_inp=1, dpmt_outp=5,
                                    ideal_match_vector=RoutedElement.xavier_vector(Settings.full_dim(), rng))
                      for kind in kinds]
        trainer = RouterTrainer(learning_rate=0.05)
        scores = {c.name: [] for c in candidates}
        for _ in range(300):
            k = dice.randrange(3)
            trace = env.choose(task(kinds[k]), candidates, rng=dice)
            score = 0.95 if trace.winner.name == kinds[k][4] else 0.60
            # Оракул знает, кто прав, то есть играет роль человека, а не критика: автоотзыв
            # входит в обучение с весом 0,25, и та же выборка учила бы вчетверо медленнее
            trainer.train(trace, Feedback(FeedbackType.HUMAN, score))
            own = scores[trace.winner.name]
            own.append(score)
            trace.winner.experience = len(own)
            trace.winner.score_variance = 0.0 if len(own) < 2 else float(np.var(own, ddof=1))
        return sum(env.get_top_k(task(kinds[k]), candidates)[0][1].name == kinds[k][4] for k in range(3))

    results = [run(seed) for seed in range(1, 21)]
    assert sum(r == 3 for r in results) >= 12


def test_auto_feedback_moves_weights_weaker_than_human():
    """Автоотзыв входит в обучение с весом AUTO_FEEDBACK_WEIGHT: шум разбора по пунктам не должен
    иметь тот же голос, что оценка человека, но и молчать до первого человека роутер не должен."""
    features = InputFeaturesService.get_features("Напиши научный обзор")
    features.input_specifications = spec(Style.SCIENTIFIC, 4000, 0.7, 0.9)
    vector = features.feature_vector()

    def shift(ftype):
        good = RoutedElement("Подходящая", ideal_match_vector=np.zeros(Settings.full_dim()))
        bad = RoutedElement("Неподходящая", ideal_match_vector=np.zeros(Settings.full_dim()))
        trace = Tracert(winner=good, top_k_elements=[good, bad], input_feature_vector=vector)
        RouterTrainer(learning_rate=0.05).train(trace, Feedback(ftype, 1.0))
        return float(good.ideal_match_vector @ vector - bad.ideal_match_vector @ vector)

    human, auto = shift(FeedbackType.HUMAN), shift(FeedbackType.AUTO)
    assert auto > 0
    assert abs(auto - human * RouterTrainer.AUTO_FEEDBACK_WEIGHT) < 1e-9


# Те же проверки, что в C#-тестах обучения (TrainingTests.cs)


def scene_task():
    return InputFeaturesService.get_features("Разбери задачу и предложи решение.")


def rated(task_features, name, quality):
    return RoutedElement(name, tps=100, dpmt_inp=1, dpmt_outp=5, ideal_match_vector=task_features.feature_vector() * quality)


def trace_of(task_features, winner, *rivals, exploration=False):
    return Tracert(winner=winner, top_k_elements=[winner, *rivals],
                   input_feature_vector=task_features.feature_vector(), is_exploration=exploration)


LIKE = Feedback(FeedbackType.HUMAN, 1.0)
DISLIKE = Feedback(FeedbackType.HUMAN, 0.0)


def test_gradient_goes_only_to_the_winner():
    """Соперников на ходе не пробовали: их векторы неподвижны, двигается только победитель, и его
    вектор подменяется целиком, а прежний объект не тронут."""
    task_features = scene_task()
    winner, rival = rated(task_features, "победитель", 0.5), rated(task_features, "соперник", 0.6)
    rival_before = rival.ideal_match_vector.copy()
    winner_before = winner.ideal_match_vector
    values = winner_before.copy()

    loss = RouterTrainer(0.05).train(trace_of(task_features, winner, rival), LIKE)

    assert loss > 0
    assert np.array_equal(rival.ideal_match_vector, rival_before)
    assert winner.get_quality_score(task_features.feature_vector()) > 0.5
    assert winner.ideal_match_vector is not winner_before and np.array_equal(winner_before, values)


def test_dislike_moves_the_winner_down():
    task_features = scene_task()
    winner = rated(task_features, "победитель", 0.6)
    RouterTrainer(0.05).train(trace_of(task_features, winner, rated(task_features, "соперник", 0.55)), DISLIKE)
    assert winner.get_quality_score(task_features.feature_vector()) < 0.6


def test_penalty_is_averaged_over_rivals():
    """Ход с тремя одинаковыми соперниками учит так же, как с одним."""
    task_features = scene_task()
    one, three = rated(task_features, "w1", 0.5), rated(task_features, "w3", 0.5)
    trainer = RouterTrainer(0.05)
    single = trainer.train(trace_of(task_features, one, rated(task_features, "r", 0.6)), LIKE)
    triple = trainer.train(trace_of(task_features, three, *(rated(task_features, f"r{i}", 0.6) for i in range(3))), LIKE)
    assert single == pytest.approx(triple, abs=1e-12)
    assert np.allclose(one.ideal_match_vector, three.ideal_match_vector, atol=1e-12)


def test_exploration_rounds_can_be_weighted():
    task_features = scene_task()
    winner = rated(task_features, "победитель", 0.5)
    before = winner.ideal_match_vector.copy()
    RouterTrainer(0.05, exploration_weight=0).train(
        trace_of(task_features, winner, rated(task_features, "соперник", 0.6), exploration=True), LIKE)
    assert np.array_equal(before, winner.ideal_match_vector)


def test_weighted_fit_reduces_to_the_plain_one():
    """Равные веса дают то же, что подгонка без весов; больший вес тянет прогноз к своей точке."""
    from fai_router.training import quality_prior

    a = InputFeaturesService.get_features("Напиши код на Python для сортировки списка").feature_vector()
    b = InputFeaturesService.get_features("Напиши эссе о погоде").feature_vector()
    points = [(a, 0.9), (b, 0.1)]
    plain = quality_prior.fit(points, None, None)
    equal = quality_prior.fit(points, [0.5, 0.5], None)
    heavy = quality_prior.fit(points, [10, 1], None)
    assert np.allclose(plain, equal, atol=1e-12)
    assert abs(heavy @ a - 0.9) < abs(plain @ a - 0.9)


def profile_mean():
    from fai_router.training import benchmark_prior

    return np.mean([task.feature_vector() for tasks in benchmark_prior.PROFILES.values() for task in tasks], axis=0)


def test_prior_keeps_its_own_task_mean():
    """Начальный вектор строится и работает в одном пространстве: среднее задач принадлежит
    вектору, и общее Settings.task_mean его не меняет. Прежде среднее приходило после построения
    приора, и прогноз на собственных замерах портился с 0,09 до 0,48."""
    from fai_router import benchmarks, catalog
    from fai_router.enums import Capability
    from fai_router.training import benchmark_prior

    snapshot = benchmarks.default_snapshot()
    model = catalog.ModelInfo("anthropic/claude-opus-4.7", "Opus", 5, 25, 200_000, 32_000, Capability.ALL)
    mean = profile_mean()
    element = catalog.create_element(model, None, snapshot, mean)
    typical = benchmark_prior.typical_task().feature_vector()
    before = element.get_quality_score(typical)

    Settings.task_mean = np.zeros(Settings.full_dim()) + 0.3
    assert element.get_quality_score(typical) == pytest.approx(before, abs=1e-12)
    assert element.task_mean is mean
    assert element.prior_experience == Settings.PRIOR_EXPERIENCE


def test_centering_by_profile_mean_hurts_the_prior():
    """Почему фабрика не вычитает среднее: вычитание с возвратом длины к единице отнимает у
    линейного прогноза общий уровень модели, и приор хуже повторяет собственные замеры. Числа те же,
    что в C#-тесте Centering_by_profile_mean_hurts_the_prior: на встроенном снимке (48 популярных
    моделей) ошибка 0,086 без среднего и 0,242 со средним профилей."""
    from fai_router import benchmarks, catalog
    from fai_router.training import benchmark_prior

    snapshot = benchmarks.default_snapshot()
    models = [model_id for model_id in catalog.popular_models() if benchmark_prior.measurements(snapshot, model_id)]

    def error(mean):
        total = 0.0
        for model_id in models:
            prior = benchmark_prior.vector(snapshot, model_id, mean)
            points = benchmark_prior.measurements(snapshot, model_id)
            total += np.mean([abs(Settings.center(task, mean) @ prior - quality) for task, quality in points])
        return total / len(models)

    plain, centered = error(None), error(profile_mean())
    assert len(models) == 48
    assert plain == pytest.approx(0.086, abs=5e-4)
    assert centered == pytest.approx(0.242, abs=5e-4)
