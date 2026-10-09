using AI.DataStructs.Algebraic;
using FAI.Router.RoutedElements;
using FAI.Router.Training;
using static FAI.Router.Tests.Scene;

namespace FAI.Router.Tests;

/// <summary>Выбор кандидата: жребий, температура, метрика R, планка и отсев по объему</summary>
public class SelectionTests
{
    private static readonly RouteWeights QualityOnly = new(1, 0, 0, 0);
    private static readonly RouteWeights PriceOnly = new(0, 1, 0, 0);

    /// <summary>Максимум берется настоящий, а не первый: иначе экспонента улетала в бесконечность и жребий ломался</summary>
    [Fact]
    public void Sample_uses_the_true_maximum()
    {
        var features = Features();
        var low = Element(features, "низкий", 0.1, 1);
        var high = Element(features, "высокий", 0.9, 1);
        Random random = new(7);

        for (int i = 0; i < 20; i++)
            Assert.Equal(1, Env.Sample([(0, low), (50, high)], new RouteWeights(1, 0, 0, 30), random));
    }

    [Fact]
    public void Sample_is_reproducible_with_a_seed()
    {
        var features = Features();
        List<(double, BaseRoutedElement)> group =
            [(0.1, Element(features, "a", 0.5, 1, experience: 0)), (0.0, Element(features, "b", 0.5, 1, experience: 0)), (-0.1, Element(features, "c", 0.5, 1, experience: 0))];

        Random one = new(42), two = new(42);

        int[] a = [.. Enumerable.Range(0, 30).Select(_ => Env.Sample(group, null, one))];
        int[] b = [.. Enumerable.Range(0, 30).Select(_ => Env.Sample(group, null, two))];

        Assert.Equal(a, b);
        Assert.True(a.Distinct().Count() > 1, "у новичков жребий должен давать разных победителей");
    }

    /// <summary>Начальные веса по рейтингам стоят опыта: такая группа остывает быстрее совсем незнакомой</summary>
    [Fact]
    public void Prior_experience_cools_the_group()
    {
        var features = Features();
        var cold = Element(features, "холодный", 0.5, 1, experience: 0);
        var rated = Element(features, "с рейтингом", 0.5, 1, experience: 0);
        rated.PriorExperience = Settings.PriorExperience;

        Assert.True(Env.Temperature([rated]) < Env.Temperature([cold]));
    }

    /// <summary>NaN в векторе одного кандидата не отравляет выбор: он получает худшее качество группы</summary>
    [Fact]
    public void NaN_forecast_gets_the_worst_quality_not_poisons_the_group()
    {
        var features = Features();
        var broken = Element(features, "сломанный", 0.9, 1);
        broken.IdealMatchVector[0] = double.NaN;

        var top = Env.GetTopK(features, [broken, Element(features, "a", 0.3, 1), Element(features, "b", 0.6, 1)], weights: QualityOnly);

        Assert.All(top, item => Assert.True(double.IsFinite(item.Score)));
        Assert.Equal("b", top[0].Element.Name);
    }

    [Fact]
    public void Non_finite_numbers_are_rejected_on_the_element()
    {
        BaseRoutedElement element = new();

        Assert.Throws<ArgumentException>(() => element.IdealMatchVector = new Vector([double.NaN]));
        Assert.Throws<ArgumentOutOfRangeException>(() => element.TPS = double.PositiveInfinity);
        Assert.Throws<ArgumentOutOfRangeException>(() => element.CostRatio = -1);
        Assert.Throws<ArgumentOutOfRangeException>(() => element.DPMTInp = double.NaN);
    }

    /// <summary>Неизвестная цена (минус единица у openrouter/auto) это худшая цена группы, а не NaN или «бесплатно»</summary>
    [Fact]
    public void Unknown_price_ranks_as_the_most_expensive()
    {
        var features = Features();
        var unknown = Element(features, "неизвестная", 0.5, -1);
        var paid = Element(features, "платная", 0.5, 10);

        Assert.False(unknown.HasKnownPrice);

        var top = Env.GetTopK(features, [unknown, paid], weights: PriceOnly);

        Assert.Equal("платная", top[0].Element.Name);
        Assert.All(top, item => Assert.True(double.IsFinite(item.Score)));
    }

    /// <summary>Ничьи разбираются по прогнозу, затем по имени: порядок каталога выбор не решает</summary>
    [Fact]
    public void Ties_are_broken_by_quality_then_name()
    {
        var features = Features();
        string[] forward = [.. Env.GetTopK(features, [Element(features, "b", 0.5, 1), Element(features, "a", 0.5, 1)]).Select(item => item.Element.Name!)];
        string[] backward = [.. Env.GetTopK(features, [Element(features, "a", 0.5, 1), Element(features, "b", 0.5, 1)]).Select(item => item.Element.Name!)];

        Assert.Equal(["a", "b"], forward);
        Assert.Equal(forward, backward);
    }

    /// <summary>
    /// Профиль «только качество» с планкой: прежде вес качества среди прошедших обнулялся, и порядок
    /// задавал каталог
    /// </summary>
    [Fact]
    public void Quality_only_profile_with_a_bar_orders_by_quality()
    {
        var features = Features();
        SufficiencyBar everyone = new(0.5, new Calibration(0, 10), PriorRate: 0.99);

        var chosen = Env.GetSufficient(features, [Element(features, "слабая", 0.4, 1), Element(features, "сильная", 0.9, 1)], everyone, weights: QualityOnly);

        Assert.True(chosen.Reached);
        Assert.Equal("сильная", chosen.Top[0].Element.Name);
    }

    /// <summary>
    /// Холодный старт под планкой: новичок получает свой откалиброванный прогноз, стянутый к доле
    /// лайков. Прежде все новички получали одну долю лайков, и ниже планки не проходил ни один
    /// </summary>
    [Fact]
    public void Cold_candidates_differ_by_their_forecast()
    {
        var features = Features();
        SufficiencyBar bar = new(0.55, new Calibration(10, -5), PriorRate: 0.5);
        var strong = Element(features, "сильный", 0.95, 1, experience: 0);
        var weak = Element(features, "слабый", 0.2, 0.1, experience: 0);

        Assert.True(bar.Sufficiency(0, 0.95) > bar.Sufficiency(0, 0.2));

        var chosen = Env.GetSufficient(features, [weak, strong], bar);

        Assert.True(chosen.Reached);
        Assert.Equal(["сильный"], chosen.Top.Select(item => item.Element.Name));
    }

    /// <summary>При недоборе ход разыгрывается среди сильнейших: иначе первый в списке получал все ходы</summary>
    [Fact]
    public void Shortfall_draws_lots_among_the_strongest()
    {
        var features = Features();
        SufficiencyBar unreachable = new(0.99, new Calibration(10, -5), PriorRate: 0.3);
        BaseRoutedElement[] cold = [.. Enumerable.Range(0, 4).Select(i => Element(features, $"новичок{i}", 0.5, 1, experience: 0))];
        Random random = new(3);

        HashSet<string> winners = [.. Enumerable.Range(0, 60).Select(_ => Env.ChooseSufficient(features, cold, unreachable, random: random).Winner.Name!)];

        Assert.True(winners.Count > 1);
        Assert.False(Env.ChooseSufficient(features, cold, unreachable, random: random).BarReached);
    }

    /// <summary>Объем не вошел ни в кого: ход отдается кандидату с наибольшим окном, недобор помечен</summary>
    [Fact]
    public void Context_shortfall_takes_the_largest_window()
    {
        var features = Features();
        features.InputLen = 1_000_000;
        var small = Element(features, "малое окно", 0.9, 1);
        var large = Element(features, "большое окно", 0.1, 1);
        small.ContextWindow = 8_000;
        large.ContextWindow = 200_000;

        var trace = Env.Choose(features, [small, large]);

        Assert.Equal("большое окно", trace.Winner.Name);
        Assert.True(trace.ContextShortfall);

        features.InputLen = 100;
        Assert.False(Env.Choose(features, [small, large], weights: QualityOnly).ContextShortfall);
    }

    /// <summary>Окно проверяется по всему входу вместе с ответом, а не только по объему ответа</summary>
    [Fact]
    public void Window_is_checked_against_the_whole_dialog()
    {
        var features = Features();
        var element = Element(features, "a", 0.5, 1);
        element.ContextWindow = 10_000;

        features.InputLen = 9_000;
        features.LenAnswer = 2_000;

        Assert.True(element.Supports(features.InputSpecifications));
        Assert.False(element.Supports(features));
    }
}
