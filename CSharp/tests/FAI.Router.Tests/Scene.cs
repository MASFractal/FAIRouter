using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using FAI.Router.Services;
using Microsoft.Data.Sqlite;

// Часть тестов трогает общие настройки процесса (Settings.TaskMean, Settings.LLM): параллельный
// запуск сделал бы их результат зависящим от соседей
[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace FAI.Router.Tests;

/// <summary>Общие заготовки тестов: задача, кандидаты с заданным прогнозом и временная база</summary>
internal static class Scene
{
    /// <summary>Признаки обычной задачи без обращения к модели</summary>
    public static InputFeatures Features(string text = "Разбери задачу и предложи решение.") => InputFeaturesService.GetFeatures(text);

    /// <summary>
    /// Кандидат с заданным прогнозом: вектор сонаправлен задаче, и при пустом среднем прогноз равен
    /// множителю
    /// </summary>
    public static BaseRoutedElement Element(InputFeatures features, string name, double quality, double price,
        double experience = 1000, double tps = 100, double variance = 0.01) =>
        new()
        {
            Name = name,
            TPS = tps,
            DPMTInp = price,
            DPMTOutp = price * 5,
            IdealMatchVector = features.FeatureVector * quality,
            Experience = experience,
            ScoreVariance = variance,
        };

    /// <summary>Трассировка хода с заданным победителем и соперниками</summary>
    public static Tracert Trace(InputFeatures features, BaseRoutedElement winner, params BaseRoutedElement[] rivals) => new()
    {
        Winner = winner,
        TopKElements = [winner, .. rivals],
        InputFeatureVector = features.FeatureVector,
        RequestedSpec = features.InputSpecifications,
    };

    /// <summary>Копия вектора значениями, чтобы сравнивать до и после</summary>
    public static double[] Values(Vector vector) => [.. vector];
}

/// <summary>Временный файл базы, удаляется вместе с WAL</summary>
internal sealed class TempDatabase : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"fai-router-test-{Guid.NewGuid():N}.db");

    public void Dispose()
    {
        SqliteConnection.ClearAllPools();

        foreach (string file in new[] { Path, Path + "-wal", Path + "-shm" })
            if (File.Exists(file))
                File.Delete(file);
    }
}

/// <summary>Распознаватель задания без модели: отдает готовую спецификацию</summary>
internal sealed class FixedSpecs(Specifications specifications) : ISpecService
{
    public int Calls { get; private set; }

    public Task<Specifications> GetSpecificationsAsync(string text, CancellationToken cancellationToken = default)
    {
        Calls++;
        return Task.FromResult(specifications);
    }
}
