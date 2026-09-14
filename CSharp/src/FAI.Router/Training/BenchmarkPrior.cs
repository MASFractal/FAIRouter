using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using AI.DataStructs.Algebraic;
using FAI.Router.Catalog;
using FAI.Router.JudgeLogic;
using FAI.Router.RotationTracking;

namespace FAI.Router.Training;

/// <summary>
/// Начальный вектор кандидата из рейтингов арены и Artificial Analysis.
/// </summary>
/// <remarks>
/// У рейтингов есть оценки моделей по сериям (категории арены, отраслевые индексы, фактология по
/// областям), а у роутера есть механизм начальных весов из замеров по типам задач
/// (<see cref="QualityPrior.FromMeasurements"/>). Здесь серии превращаются в типы задач: каждой
/// серии соответствует профиль признаков, а оценка модели в ней становится качеством на этом
/// профиле. Профили лежат в файле benchmark_profiles.json, общем с версией на Python: он встроен в
/// сборку ресурсом, и расходиться профилям негде.
/// </remarks>
public static class BenchmarkPrior
{
    /// <summary>
    /// Верх поправки цены из доли рассуждений: доля бывает под 0,99, а один такой замер не повод
    /// считать модель стократно дороже
    /// </summary>
    public const double MaxCostRatio = 5;

    private const string ResourceName = "FAI.Router.benchmark_profiles.json";

    // Имена полей в файле змеиные, перечисления записаны именами, как в версии на Python
    private static readonly JsonSerializerOptions SpecOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Converters = { new JsonStringEnumConverter() }
    };

    private static readonly JsonObject Source = LoadSource();

    /// <summary>Профили по ключам серий, в порядке файла</summary>
    public static readonly IReadOnlyDictionary<string, IReadOnlyList<InputFeatures>> Profiles = BuildProfiles();

    /// <summary>Опорная задача: обычный развернутый ответ пользователю</summary>
    public static InputFeatures TypicalTask() => Task(new JsonObject());

    /// <summary>Пары «вектор задачи, качество» для FromMeasurements по всем сериям, где модель есть</summary>
    public static IReadOnlyList<(Vector Task, double Quality)> Measurements(BenchmarkSnapshot snapshot, string openRouterId)
    {
        List<(Vector, double)> pairs = [];

        foreach ((string key, IReadOnlyList<InputFeatures> tasks) in Profiles)
        {
            double? quality = snapshot.Quality(key, openRouterId);

            if (quality is null)
                continue;

            pairs.AddRange(tasks.Select(task => (task.FeatureVector, quality.Value)));
        }

        return pairs;
    }

    /// <summary>Начальный вектор кандидата по рейтингам; <c>null</c>, если модели нет ни в одной серии</summary>
    public static Vector? GetVector(BenchmarkSnapshot snapshot, string openRouterId)
    {
        IReadOnlyList<(Vector Task, double Quality)> pairs = Measurements(snapshot, openRouterId);

        return pairs.Count == 0 ? null : QualityPrior.FromMeasurements(pairs);
    }

    /// <summary>Скорость модели по замеру Artificial Analysis, токенов в секунду; нет замера, значит <c>null</c></summary>
    public static double? TokensPerSecond(BenchmarkSnapshot snapshot, string openRouterId) =>
        snapshot.Value("aa:speed", openRouterId) is > 0 and var speed ? speed : null;

    /// <summary>
    /// Начальная поправка цены: задача обходится дороже прайса ответа на долю рассуждений. Среднее
    /// по отраслевым индексам, где модель есть; нет ни в одном, значит <c>null</c>.
    /// </summary>
    public static double? CostRatio(BenchmarkSnapshot snapshot, string openRouterId)
    {
        double[] shares =
        [
            .. AnalysisLeaderboard.Indices
                .Select(index => snapshot.Value($"aa:reasoning-share/{index}", openRouterId))
                .Where(share => share is not null)
                .Select(share => share!.Value)
        ];

        if (shares.Length == 0)
            return null;

        double share = shares.Average();

        return Math.Clamp(1.0 / Math.Max(1.0 - share, 1e-9), 1.0, MaxCostRatio);
    }

    /// <summary>Задача по записи профиля: base.spec, затем шаблон, затем spec профиля</summary>
    public static InputFeatures Task(JsonObject item)
    {
        JsonObject baseTask = Source["base"]!.AsObject();
        JsonObject spec = baseTask["spec"]!.DeepClone().AsObject();

        if (item["template"]?.GetValue<string>() is { } template)
            Merge(spec, Source["templates"]![template]!.AsObject());

        if (item["spec"] is JsonObject overrides)
            Merge(spec, overrides);

        return new InputFeatures
        {
            InputLen = (item["input_len"] ?? baseTask["input_len"])!.GetValue<double>(),
            LenAnswer = (item["len_answer"] ?? baseTask["len_answer"])!.GetValue<double>(),
            TurnCount = (item["turn_count"] ?? baseTask["turn_count"])!.GetValue<int>(),
            InputSpecifications = spec.Deserialize<Specifications>(SpecOptions) ?? new Specifications(),
        };
    }

    private static void Merge(JsonObject target, JsonObject source)
    {
        foreach ((string key, JsonNode? value) in source)
            target[key] = value?.DeepClone();
    }

    private static JsonObject LoadSource()
    {
        using Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException($"В сборке нет ресурса {ResourceName}.");

        return JsonNode.Parse(stream)!.AsObject();
    }

    private static Dictionary<string, IReadOnlyList<InputFeatures>> BuildProfiles()
    {
        Dictionary<string, IReadOnlyList<InputFeatures>> profiles = [];

        foreach ((string key, JsonNode? items) in Source["profiles"]!.AsObject())
            profiles[key] = [.. items!.AsArray().Select(item => Task(item!.AsObject()))];

        return profiles;
    }
}
