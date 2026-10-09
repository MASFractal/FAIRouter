using AI.DataStructs.Algebraic;
using FAI.Router.Enums;
using FAI.Router.Persistence;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using Microsoft.Data.Sqlite;
using static FAI.Router.Tests.Scene;

namespace FAI.Router.Tests;

/// <summary>Журнал и веса в SQLite: миграция схемы, порядок обучения, опыт, калибровка, загрузка</summary>
public class PersistenceTests
{
    private static readonly Feedback Human = new() { FType = FeedbackType.Human, FeadbackScore = 1.0 };

    /// <summary>База прежней версии доводится до новой схемы без потери ходов; журнал в режиме WAL</summary>
    [Fact]
    public void Old_database_is_migrated_in_place()
    {
        using TempDatabase database = new();

        using (SqliteConnection connection = new($"Data Source={database.Path}"))
        {
            connection.Open();
            using SqliteCommand command = connection.CreateCommand();
            command.CommandText =
                """
                CREATE TABLE rounds (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, prompt TEXT,
                    winner TEXT NOT NULL, rivals_json TEXT NOT NULL, features_json TEXT NOT NULL, score REAL NOT NULL DEFAULT 0,
                    is_exploration INTEGER NOT NULL DEFAULT 0, requested_json TEXT, actual_json TEXT, feedback_type INTEGER, feedback_score REAL);
                CREATE TABLE element_vectors (name TEXT PRIMARY KEY, dimension INTEGER NOT NULL, values_json TEXT NOT NULL);
                INSERT INTO rounds (created_at, winner, rivals_json, features_json, feedback_type, feedback_score)
                VALUES ('2026-01-01T00:00:00.0000000Z', 'a', '["b"]', '[1,0]', 1, 1.0);
                """;
            command.ExecuteNonQuery();
        }

        var features = Features();
        SqliteTraceStore store = new(database.Path);
        TrainingRound round = Assert.Single(store.ReadRated([Element(features, "a", 0.5, 1), Element(features, "b", 0.5, 1)]));

        Assert.Equal("a", round.Trace.Winner.Name);
        Assert.Null(round.Trace.Forecast);
        Assert.Equal(2L, Scalar(database.Path, "PRAGMA user_version"));
        Assert.Equal("wal", Scalar(database.Path, "PRAGMA journal_mode"));
        Assert.Equal(1L, Scalar(database.Path, "SELECT COUNT(*) FROM pragma_table_info('element_vectors') WHERE name = 'mean_json'"));
    }

    /// <summary>Обучение идет по порядку записи, только по необученным, а снятый соперник выпадает из пары, не из выборки</summary>
    [Fact]
    public void Rated_rounds_come_in_order_and_once()
    {
        using TempDatabase database = new();
        var features = Features();
        var a = Element(features, "a", 0.5, 1);
        var b = Element(features, "b", 0.5, 1);
        var gone = Element(features, "снятая", 0.5, 1);
        SqliteTraceStore store = new(database.Path);

        long first = store.Append(Trace(features, a, b, gone));
        long second = store.Append(Trace(features, b, a));
        store.SetFeedback(first, Human);
        store.SetFeedback(second, Human);

        IReadOnlyList<TrainingRound> rounds = store.ReadRated([a, b, a]);

        Assert.Equal([first, second], rounds.Select(round => round.Id));
        Assert.Equal(["a", "b"], rounds[0].Trace.TopKElements.Select(element => element.Name));

        store.MarkTrained([first]);
        Assert.Equal([second], store.ReadRated([a, b], untrainedOnly: true).Select(round => round.Id));

        store.SetFeedback(first, new Feedback { FType = FeedbackType.Human, FeadbackScore = 0 });
        Assert.Equal(2, store.ReadRated([a, b], untrainedOnly: true).Count);
    }

    /// <summary>Автоотзывы входят в опыт с весом; два одинаковых отзыва не дают нулевой дисперсии</summary>
    [Fact]
    public void Statistics_count_auto_feedback_and_shrink_variance()
    {
        using TempDatabase database = new();
        var features = Features();
        var a = Element(features, "a", 0.5, 1, experience: 0);
        var b = Element(features, "b", 0.5, 1, experience: 0);
        SqliteTraceStore store = new(database.Path);

        store.SetFeedback(store.Append(Trace(features, a, b)), Human);

        for (int i = 0; i < 4; i++)
            store.SetFeedback(store.Append(Trace(features, a, b)), new Feedback { FType = FeedbackType.Auto, FeadbackScore = 0.8 });

        store.SetFeedback(store.Append(Trace(features, b, a)), Human);
        store.SetFeedback(store.Append(Trace(features, b, a)), Human);

        store.LoadStatistics([a, b, b]);

        Assert.Equal(1 + 4 * Settings.AutoFeedbackWeight, a.Experience, 9);
        Assert.Equal(2, b.Experience, 9);
        Assert.True(b.ScoreVariance > 0);
    }

    /// <summary>Калибровка берет прогноз в момент выбора и только человеческие отзывы</summary>
    [Fact]
    public void Calibration_uses_forecast_at_selection_time()
    {
        using TempDatabase database = new();
        var features = Features();
        var a = Element(features, "a", 0.5, 1);
        var b = Element(features, "b", 0.5, 1);
        SqliteTraceStore store = new(database.Path);

        var human = Trace(features, a, b);
        human.Forecast = 0.123;
        store.SetFeedback(store.Append(human), Human);

        var auto = Trace(features, a, b);
        auto.Forecast = 0.9;
        store.SetFeedback(store.Append(auto), new Feedback { FType = FeedbackType.Auto, FeadbackScore = 0.2 });

        a.IdealMatchVector = features.FeatureVector * 0.99;

        var pair = Assert.Single(store.ReadCalibration([a, b]));
        Assert.Equal((0.123, 1.0), pair);
    }

    [Fact]
    public void Feature_mean_reads_only_recent_rounds_and_purge_forgets_old_ones()
    {
        using TempDatabase database = new();
        var features = Features();
        var a = Element(features, "a", 0.5, 1);
        SqliteTraceStore store = new(database.Path);

        for (int i = 0; i < 3; i++)
            store.Append(Trace(features, a), prompt: "текст");

        Assert.NotNull(store.GetFeatureMean(limit: 2));
        Assert.Equal(3, store.Purge(DateTimeOffset.UtcNow.AddMinutes(1), promptsOnly: true));
        Assert.Equal(0, store.Purge(DateTimeOffset.UtcNow.AddDays(-1)));
        Assert.Equal(3, store.Purge(DateTimeOffset.UtcNow.AddMinutes(1)));
        Assert.Equal((0, 0), store.Count());
    }

    /// <summary>Вектор чужой размерности пропускается, остальные загружаются; среднее при векторе переживает сохранение</summary>
    [Fact]
    public void Wrong_dimension_vector_is_skipped_not_fatal()
    {
        using TempDatabase database = new();
        var features = Features();
        var good = Element(features, "good", 0.7, 1);
        good.TaskMean = new Vector(Settings.VectorDim) + 0.01;
        SqliteWeightsStore store = new(database.Path);

        store.Save([good]);
        Execute(database.Path, "INSERT INTO element_vectors (name, dimension, values_json) VALUES ('bad', 2, '[1,2]')");
        store.SaveTaskMean(new Vector([1.0, 2.0]));

        var freshGood = Element(features, "good", 0.1, 1);
        var freshBad = Element(features, "bad", 0.1, 1);
        double[] badBefore = Values(freshBad.IdealMatchVector);

        Assert.Equal(["good"], store.LoadVectors([freshGood, freshBad]));
        Assert.Equal(Values(good.IdealMatchVector), Values(freshGood.IdealMatchVector));
        Assert.Equal(Values(good.TaskMean), Values(freshGood.TaskMean!));
        Assert.Equal(badBefore, Values(freshBad.IdealMatchVector));
        Assert.Null(store.LoadTaskMean());
    }

    /// <summary>
    /// Роутер учит каждый ход один раз, сохраняет только обученные векторы и не трогает общее среднее
    /// задач: прежде Load и первый Train задавали Settings.TaskMean, и прогноз необученных кандидатов
    /// пересчитывался в чужом пространстве
    /// </summary>
    [Fact]
    public void Router_trains_once_saves_only_trained_and_keeps_global_mean()
    {
        using TempDatabase database = new();
        var features = Features();
        Vector? globalMean = Settings.TaskMean;

        // Прежняя база с общим средним: оно не должно сдвинуть прогноз необученного кандидата
        new SqliteWeightsStore(database.Path).SaveTaskMean(new Vector(Settings.VectorDim) + 0.05);

        BaseRoutedElement[] Candidates() => [Element(features, "a", 0.5, 1), Element(features, "b", 0.6, 1), Element(features, "c", 0.4, 1)];

        BaseRoutedElement[] first = Candidates();
        double untrainedForecast = first[2].GetQualityScore(features.FeatureVector);
        FaiRouter router = new(first, (_, _) => Task.FromResult("ответ"), database.Path, measure: false);

        Assert.Equal(untrainedForecast, first[2].GetQualityScore(features.FeatureVector), 12);
        Assert.Same(globalMean, Settings.TaskMean);

        long round = router.Traces!.Append(Trace(features, first[0], first[1]));
        router.Feedback(round, 1.0);

        TrainingLoss loss = router.Train();
        Assert.Equal(1, loss.Rounds);
        Assert.True(loss.Router > 0);
        Assert.Equal(0, router.Train().Rounds);
        Assert.Same(globalMean, Settings.TaskMean);

        router.Save();

        BaseRoutedElement[] second = Candidates();
        _ = new FaiRouter(second, (_, _) => Task.FromResult("ответ"), database.Path, measure: false);

        Assert.Equal(Values(first[0].IdealMatchVector), Values(second[0].IdealMatchVector));
        Assert.Equal(Values(Candidates()[1].IdealMatchVector), Values(second[1].IdealMatchVector));
        Assert.Equal(1L, Scalar(database.Path, "SELECT COUNT(*) FROM element_vectors"));
        Assert.Equal(0, new FaiRouter(Candidates(), (_, _) => Task.FromResult("ответ"), database.Path, measure: false).Train().Rounds);
    }

    /// <summary>Явный множитель температуры берется как есть, даже совпадая с общим; профили получают множитель роутера</summary>
    [Fact]
    public void Explicit_temperature_scale_is_kept()
    {
        FaiRouter router = new([Element(Features(), "a", 0.5, 1)], (_, _) => Task.FromResult("ответ"), measure: false, temperatureScale: 10);

        Assert.Equal(10, router.WeightsFor(RouteWeights.Quality)!.Value.TemperatureScale);
        Assert.Equal(Settings.TemperatureScale, router.WeightsFor(new RouteWeights(0.5, 0.25, 0.25, Settings.TemperatureScale))!.Value.TemperatureScale);
        Assert.Throws<ArgumentOutOfRangeException>(() => Settings.FeaturesDim = 6);
    }

    private static object? Scalar(string path, string sql)
    {
        using SqliteConnection connection = new($"Data Source={path}");
        connection.Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = sql;

        return command.ExecuteScalar();
    }

    private static void Execute(string path, string sql)
    {
        using SqliteConnection connection = new($"Data Source={path}");
        connection.Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = sql;
        command.ExecuteNonQuery();
    }
}
