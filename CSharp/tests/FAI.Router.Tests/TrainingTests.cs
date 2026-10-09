using AI.DataStructs.Algebraic;
using FAI.Router.Catalog;
using FAI.Router.Enums;
using FAI.Router.RotationTracking;
using FAI.Router.Training;
using Xunit.Abstractions;
using static FAI.Router.Tests.Scene;

namespace FAI.Router.Tests;

/// <summary>Обучение роутера, калибровка планки и начальные веса по рейтингам</summary>
public class TrainingTests(ITestOutputHelper output)
{
    private static readonly Feedback Like = new() { FType = FeedbackType.Human, FeadbackScore = 1.0 };
    private static readonly Feedback Dislike = new() { FType = FeedbackType.Human, FeadbackScore = 0.0 };

    /// <summary>Соперников на ходе не пробовали: их векторы неподвижны, двигается только победитель</summary>
    [Fact]
    public void Gradient_goes_only_to_the_winner()
    {
        var features = Features();
        var winner = Element(features, "победитель", 0.5, 1);
        var rival = Element(features, "соперник", 0.6, 1);
        double[] rivalBefore = Values(rival.IdealMatchVector);
        Vector winnerBefore = winner.IdealMatchVector;
        double[] winnerValues = Values(winnerBefore);

        double loss = new RouterTrainer(0.05f).Train(Trace(features, winner, rival), Like);

        Assert.True(loss > 0);
        Assert.Equal(rivalBefore, Values(rival.IdealMatchVector));
        Assert.True(winner.GetQualityScore(features.FeatureVector) > 0.5);

        // Вектор подменен целиком, а прежний объект не тронут: соседний выбор не видит смеси
        Assert.NotSame(winnerBefore, winner.IdealMatchVector);
        Assert.Equal(winnerValues, Values(winnerBefore));
    }

    [Fact]
    public void Dislike_moves_the_winner_down()
    {
        var features = Features();
        var winner = Element(features, "победитель", 0.6, 1);

        new RouterTrainer(0.05f).Train(Trace(features, winner, Element(features, "соперник", 0.55, 1)), Dislike);

        Assert.True(winner.GetQualityScore(features.FeatureVector) < 0.6);
    }

    /// <summary>Штраф усредняется по соперникам: ход с тремя одинаковыми соперниками учит так же, как с одним</summary>
    [Fact]
    public void Penalty_is_averaged_over_rivals()
    {
        var features = Features();
        var one = Element(features, "w1", 0.5, 1);
        var three = Element(features, "w3", 0.5, 1);
        RouterTrainer trainer = new(0.05f);

        double single = trainer.Train(Trace(features, one, Element(features, "r", 0.6, 1)), Like);
        double triple = trainer.Train(Trace(features, three, Element(features, "r1", 0.6, 1), Element(features, "r2", 0.6, 1), Element(features, "r3", 0.6, 1)), Like);

        Assert.Equal(single, triple, 12);
        Assert.Equal(Values(one.IdealMatchVector), Values(three.IdealMatchVector));
    }

    [Fact]
    public void Exploration_rounds_can_be_weighted()
    {
        var features = Features();
        var winner = Element(features, "победитель", 0.5, 1);
        double[] before = Values(winner.IdealMatchVector);
        var trace = Trace(features, winner, Element(features, "соперник", 0.6, 1));
        trace.IsExploration = true;

        new RouterTrainer(0.05f, explorationWeight: 0).Train(trace, Like);

        Assert.Equal(before, Values(winner.IdealMatchVector));
    }

    /// <summary>Числа калибровки те же, что в версии на Python: контроль шага Ньютона их не меняет</summary>
    [Fact]
    public void Calibration_matches_python_version()
    {
        (double, double)[] pairs =
        [
            (0.20, 0.0), (0.30, 0.0), (0.35, 1.0), (0.40, 0.0), (0.50, 1.0),
            (0.55, 0.0), (0.60, 1.0), (0.70, 1.0), (0.80, 1.0), (0.90, 1.0),
        ];

        var fitted = Calibration.Fit(pairs);

        Assert.Equal(3.347648495609008, fitted.A, 9);
        Assert.Equal(-1.3139404220311586, fitted.B, 9);
    }

    /// <summary>Разделимая выборка с выбросами по шкале: полный шаг Ньютона уходил бы в бесконечность</summary>
    [Fact]
    public void Calibration_stays_finite_on_separable_data()
    {
        (double, double)[] pairs = [.. Enumerable.Range(0, 40).Select(i => (i < 20 ? -50.0 + i : 50.0 + i, i < 20 ? 0.0 : 1.0))];

        var fitted = Calibration.Fit(pairs, ridge: 1e-6);

        Assert.True(double.IsFinite(fitted.A) && double.IsFinite(fitted.B));
        Assert.True(fitted.Predict(80) > fitted.Predict(-40));
    }

    /// <summary>Равные веса дают то же, что подгонка без весов; больший вес тянет прогноз к своей точке</summary>
    [Fact]
    public void Weighted_fit_reduces_to_the_plain_one()
    {
        var a = Features("Напиши код на Python для сортировки списка");
        var b = Features("Напиши эссе о погоде");
        (Vector, double)[] points = [(a.FeatureVector, 0.9), (b.FeatureVector, 0.1)];

        Vector plain = QualityPrior.Fit(points, null, null);
        Vector equal = QualityPrior.Fit(points, [0.5, 0.5], null);
        Vector heavy = QualityPrior.Fit(points, [10, 1], null);

        Assert.Equal(Values(plain), Values(equal), new ToleranceComparer(1e-12));
        Assert.True(Math.Abs(heavy.Dot(a.FeatureVector) - 0.9) < Math.Abs(plain.Dot(a.FeatureVector) - 0.9));
    }

    /// <summary>
    /// Начальный вектор строится и работает в одном пространстве: среднее задач принадлежит вектору,
    /// и общее Settings.TaskMean его не меняет. Прежде среднее приходило после построения приора, и
    /// прогноз на собственных замерах портился с 0,09 до 0,48
    /// </summary>
    [Fact]
    public void Prior_keeps_its_own_task_mean()
    {
        var snapshot = BenchmarkSnapshot.LoadEmbedded();
        ModelInfo model = new("anthropic/claude-opus-4.7", "Opus", 5, 25, 200_000, 32_000, Capability.All);
        Vector mean = ProfileMean();
        var element = ModelCatalog.CreateElement(model, null, snapshot, mean);

        double before = element.GetQualityScore(BenchmarkPrior.TypicalTask().FeatureVector);
        Vector? global = Settings.TaskMean;

        try
        {
            Settings.TaskMean = new Vector(Settings.VectorDim) + 0.3;
            Assert.Equal(before, element.GetQualityScore(BenchmarkPrior.TypicalTask().FeatureVector), 12);
        }
        finally
        {
            Settings.TaskMean = global;
        }

        Assert.Same(mean, element.TaskMean);
        Assert.Equal(Settings.PriorExperience, element.PriorExperience);
    }

    /// <summary>
    /// Почему фабрика не вычитает среднее: вычитание с возвратом длины к единице отнимает у линейного
    /// прогноза общий уровень модели, и приор хуже повторяет собственные замеры. На встроенном снимке
    /// (48 популярных моделей, 09.10.2026) ошибка 0,086 без среднего и 0,242 со средним профилей
    /// </summary>
    [Fact]
    public void Centering_by_profile_mean_hurts_the_prior()
    {
        var snapshot = BenchmarkSnapshot.LoadEmbedded();
        string[] models = [.. ModelCatalog.PopularModels().Where(id => BenchmarkPrior.Measurements(snapshot, id).Count > 0)];

        double plain = FitError(snapshot, models, null);
        double centered = FitError(snapshot, models, ProfileMean());

        output.WriteLine($"моделей {models.Length}: ошибка без среднего {plain:F3}, со средним профилей {centered:F3}");

        Assert.True(models.Length > 10);
        Assert.True(plain < 0.15);
        Assert.True(centered > plain);
    }

    private static Vector ProfileMean()
    {
        Vector[] tasks = [.. BenchmarkPrior.Profiles.Values.SelectMany(items => items).Select(task => task.FeatureVector)];
        Vector sum = new(tasks[0].Count);

        foreach (Vector task in tasks)
            sum += task;

        return sum / tasks.Length;
    }

    private static double FitError(BenchmarkSnapshot snapshot, string[] models, Vector? mean) =>
        models.Average(id =>
        {
            Vector prior = BenchmarkPrior.GetVector(snapshot, id, mean)!;

            return BenchmarkPrior.Measurements(snapshot, id).Average(point => Math.Abs(Settings.Center(point.Task, mean).Dot(prior) - point.Quality));
        });

    private sealed class ToleranceComparer(double tolerance) : IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) <= tolerance;

        public int GetHashCode(double value) => 0;
    }
}
