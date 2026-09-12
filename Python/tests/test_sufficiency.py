import numpy as np
import pytest

from fai_router import env
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import RouteWeights, SufficiencyBar
from fai_router.training import Calibration

# Те же пары и те же ожидаемые числа стоят в RouterSufficiencyTests (Mas.Core.Tests): так
# проверяется, что версии на Python и C# калибруют шаг в шаг
PAIRS = [(0.20, 0.0), (0.30, 0.0), (0.35, 1.0), (0.40, 0.0), (0.50, 1.0),
         (0.55, 0.0), (0.60, 1.0), (0.70, 1.0), (0.80, 1.0), (0.90, 1.0)]

# Заказчик, которому почти безразличны цена и время: линейная метрика отдает ему сильнейшего
QUALITY_FIRST = RouteWeights(WQ=1.0, WC=0.05, WT=0.05, temperature_scale=0)


def test_calibration_matches_csharp_version():
    fitted = Calibration.fit(PAIRS)
    assert fitted.A == pytest.approx(0.6040727816122299, abs=1e-9)
    assert fitted.B == pytest.approx(0.08713282758760345, abs=1e-9)
    assert fitted.predict(0.5) == pytest.approx(0.5960826895510254, abs=1e-9)


def test_weak_ridge_lets_the_slope_show():
    """Наклон стягивается к нулю, пока оценок мало: со слабым стягиванием зависимость видна резче."""
    assert Calibration.fit(PAIRS, ridge=0.1).A > Calibration.fit(PAIRS).A > 0


def test_unanimous_ratings_stay_finite():
    """Все оценки одинаковые: у сдвига нет конечного оптимума, привязка держит его конечным."""
    fitted = Calibration.fit([(0.5, 1.0), (0.7, 1.0), (0.9, 1.0)])
    assert np.isfinite(fitted.A) and np.isfinite(fitted.B)
    assert fitted.predict(0.7) > 0.99


def test_sufficient_pays_for_no_extra_quality():
    """Ради чего планка: линейная метрика с упором на качество берет сильную, а по принципу
    «необходимо и достаточно» хватает средней, и за лишнее качество не платят."""
    features, candidates = scene()
    assert env.get_top_k(features, candidates, weights=QUALITY_FIRST)[0][1].name == "сильная"

    chosen = env.get_sufficient(features, candidates, bar(0.7), weights=QUALITY_FIRST)
    assert chosen.reached
    assert chosen.top[0][1].name == "средняя"
    assert "слабая" not in [element.name for _, element in chosen.top]


def test_unreachable_bar_gives_the_strongest():
    """Не дотянул никто: отдается сильнейший, а решать, что сказать человеку, вызывающему."""
    features, candidates = scene()
    chosen = env.get_sufficient(features, candidates, bar(0.99), weights=QUALITY_FIRST)
    assert not chosen.reached
    assert chosen.top[0][1].name == "сильная"


def test_newcomer_gets_the_average_not_its_own_forecast():
    """Незнакомому кандидату калибровке верить рано: он получает долю лайков по всем ходам."""
    features, candidates = scene()
    newcomer = element(features, "новичок", 0.95, 0.01, experience=0)
    assert bar(0.7).sufficiency(newcomer.experience, 0.95) == pytest.approx(0.5)

    chosen = env.get_sufficient(features, [*candidates, newcomer], bar(0.7), weights=QUALITY_FIRST)
    assert "новичок" not in [candidate.name for _, candidate in chosen.top]


def test_cost_ratio_raises_cost_but_not_list_price():
    """Поправка на переделки входит в ожидаемую цену, но не в прайс: с прайсом сравнивается факт."""
    features, candidates = scene()
    mid = candidates[1]
    list_price = mid.get_list_cost(features)
    mid.cost_ratio = 3.0
    assert mid.get_list_cost(features) == list_price
    assert mid.get_cost(features) == pytest.approx(3 * list_price)


def bar(level):
    return SufficiencyBar(level, Calibration(10.0, -5.0), prior_rate=0.5)


def element(features, name, quality, price, experience=1000):
    """Кандидат с заданным прогнозом: вектор сонаправлен задаче, и прогноз равен его длине."""
    candidate = RoutedElement(name, tps=100, dpmt_inp=price, dpmt_outp=price * 5,
                              ideal_match_vector=features.feature_vector() * quality)
    candidate.experience = experience
    return candidate


def scene():
    features = InputFeaturesService.get_features("Разбери задачу и предложи решение.")
    return features, [element(features, "слабая", 0.3, 0.1),
                      element(features, "средняя", 0.6, 1.0),
                      element(features, "сильная", 0.9, 15.0)]
