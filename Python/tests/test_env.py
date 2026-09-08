import random

import numpy as np
import pytest

from fai_router import env
from fai_router.enums import Capability, Style
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import Settings
from fai_router.specifications import Specifications


def task(style=Style.SCIENTIFIC, symbols=1500, code_blocks=0):
    features = InputFeaturesService.get_features("Напиши научный обзор методов регуляризации нейросетей на 1500 знаков.")
    features.input_specifications = Specifications(style_type=style, symbol_length=symbols, word_length=symbols // 7,
                                                   section_count=3, term_density=0.7, formality_score=0.9,
                                                   code_block_count=code_blocks)
    return features


def element(name, vector, tps=100, dpmt_inp=1.0, dpmt_outp=5.0, **kwargs):
    return RoutedElement(name, tps=tps, dpmt_inp=dpmt_inp, dpmt_outp=dpmt_outp,
                         ideal_match_vector=np.array(vector, dtype=float), **kwargs)


def test_temperature_falls_with_experience_and_rises_with_newcomer():
    def group(experience):
        return [RoutedElement(f"к{i}", ideal_match_vector=np.zeros(Settings.full_dim())) for i in range(3)]

    temperatures = []
    for experience in (0, 2, 10, 200):
        members = group(experience)
        for member in members:
            member.experience, member.score_variance = experience, 0.04
        temperatures.append(env.temperature(members))
    assert temperatures == sorted(temperatures, reverse=True)

    mixed = group(0)
    for member in mixed[:2]:
        member.experience, member.score_variance = 200, 0.04
    assert env.temperature(mixed) > temperatures[-1]


def test_cost_participates_in_metric():
    """Прежняя формула цену не видела: ее вклад был в 1156 раз меньше вклада качества."""
    # Направление берется от самой задачи: тогда кратный вектор дает заведомо больший прогноз.
    # Случайный вектор для этого не годится, знак его произведения на задачу не гарантирован
    shared = task().feature_vector()
    cheap = element("дешевый", shared, dpmt_inp=0.3, dpmt_outp=2.5)
    pricey = element("дорогой", shared, dpmt_inp=15, dpmt_outp=75)
    assert env.get_top_k(task(), [pricey, cheap])[0][1] is cheap

    better = element("дорогой и лучший", shared * 3, dpmt_inp=15, dpmt_outp=75)
    assert env.get_top_k(task(), [better, cheap])[0][1] is better

    a = element("A", shared * 2)
    b = element("B", shared)
    assert env.get_top_k(task(), [b, a])[0][1] is a


def test_greedy_when_temperature_is_zero_and_spread_otherwise():
    shared = task().feature_vector()
    members = [element("лидер", shared * 2), element("второй", shared), element("третий", shared * 0.5)]
    rng = random.Random(7)

    Settings.temperature_scale = 0.0
    assert all(env.choose(task(), members, rng=rng).winner is members[0] for _ in range(50))

    Settings.temperature_scale = 20.0
    winners = [env.choose(task(), members, rng=rng).winner.name for _ in range(400)]
    assert len(set(winners)) > 1
    assert any(env.choose(task(), members, rng=rng).is_exploration for _ in range(50))


def test_capability_filter_precedes_ranking():
    shared = np.ones(Settings.full_dim())
    text_only = element("только текст", shared, tps=500, dpmt_inp=0.1, dpmt_outp=0.5, capabilities=Capability.NONE)
    coder = element("умеет код", shared, tps=50, dpmt_inp=3, dpmt_outp=15)
    kept = env.get_top_k(task(code_blocks=2), [text_only, coder])
    assert [item[1] for item in kept] == [coder]

    small = element("короткий контекст", shared, context_limit=500)
    assert env.get_top_k(task(symbols=1500), [small]) == []
    assert len(env.get_top_k(task(symbols=300), [small])) == 1

    seeing = element("зрячий", shared, capabilities=Capability.ALL)
    blind = element("слепой", shared, capabilities=Capability.CODE)
    assert [item[1] for item in env.get_top_k(task(), [blind, seeing], required=Capability.VISION)] == [seeing]

    with pytest.raises(ValueError):
        env.choose(task(code_blocks=1), [text_only])


def test_fallback_replaces_winner_with_the_one_who_delivered():
    members = [element(name, np.ones(Settings.full_dim())) for name in ("a", "b", "c")]
    trace = env.choose(task(), members)
    attempts = []

    def run(candidate):
        attempts.append(candidate.name)
        if len(attempts) < 3:
            raise RuntimeError("поставщик недоступен")
        return f"ответ от {candidate.name}"

    answer = env.execute(trace, run)
    assert answer == f"ответ от {attempts[-1]}"
    assert trace.winner.name == attempts[-1]

    with pytest.raises(RuntimeError):
        env.execute(env.choose(task(), members), lambda _: (_ for _ in ()).throw(RuntimeError("отказ")))
