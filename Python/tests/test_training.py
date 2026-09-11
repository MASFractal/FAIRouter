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
    """Как в exploration.md: с вычитанием среднего и возвратом длины 20 запусков из 20."""
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
    assert sum(r == 3 for r in results) >= 15


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
