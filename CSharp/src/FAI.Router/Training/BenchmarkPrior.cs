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
/// Начальный вектор кандидата из внешних замеров качества моделей.
/// </summary>
/// <remarks>
/// У замеров есть оценки моделей по сериям (предпочтения людей, отраслевые индексы, фактология по
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

    /// <summary>Серии с долей рассуждений в цене задачи: по ним считается начальная поправка цены</summary>
    private const string ReasoningShare = "bench:reasoning-share/";

    /// <summary>Серия скорости, токенов в секунду</summary>
    private const string Speed = "bench:speed";

    private const string ResourceName = "FAI.Router.benchmark_profiles.json";

    // Имена полей в файле змеиные, перечисления записаны именами, как в версии на Python
    private static readonly JsonSerializerOptions SpecOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Converters = { new JsonStringEnumConverter() }
    };

    private static readonly JsonObject Source = LoadSource();

    // Вектор Uniform для единицы; запись целиком, чтобы соседний поток видел либо старую пару, либо новую
    private static UniformUnit? _uniform;

    /// <summary>Профили по ключам серий, в порядке файла</summary>
    public static readonly IReadOnlyDictionary<string, IReadOnlyList<InputFeatures>> Profiles = BuildProfiles();

    /// <summary>Опорная задача: обычный развернутый ответ пользователю</summary>
    public static InputFeatures TypicalTask() => Task(new JsonObject());

    /// <summary>Пары «вектор задачи, качество» для FromMeasurements по всем сериям, где модель есть</summary>
    public static IReadOnlyList<(Vector Task, double Quality)> Measurements(BenchmarkSnapshot snapshot, string openRouterId) =>
        [.. Points(snapshot, openRouterId).Select(point => (point.Task, point.Quality))];

    /// <summary>
    /// Начальный вектор кандидата по рейтингам в пространстве общего среднего
    /// <see cref="Settings.TaskMean"/>; <c>null</c>, если модели нет ни в одной серии
    /// </summary>
    public static Vector? GetVector(BenchmarkSnapshot snapshot, string openRouterId) =>
        GetVector(snapshot, openRouterId, Settings.TaskMean);

    /// <summary>
    /// Начальный вектор кандидата по рейтингам в пространстве среднего <paramref name="mean"/>;
    /// <c>null</c>, если модели нет ни в одной серии
    /// </summary>
    /// <remarks>
    /// Точка серии весит 1/(число профилей серии): серия весит одинаково, сколько бы профилей задач за
    /// ней ни стояло. Прежде серия с четырьмя профилями тянула прогноз вчетверо сильнее серии с одним.
    /// </remarks>
    /// <param name="snapshot">Снимок замеров</param>
    /// <param name="openRouterId">Идентификатор модели в каталоге</param>
    /// <param name="mean">Среднее задач; пусто, значит без вычитания</param>
    public static Vector? GetVector(BenchmarkSnapshot snapshot, string openRouterId, Vector? mean)
    {
        (Vector Task, double Quality, double Weight)[] points = [.. Points(snapshot, openRouterId)];

        return points.Length == 0
            ? null
            : QualityPrior.Fit([.. points.Select(point => (point.Task, point.Quality))], [.. points.Select(point => point.Weight)], mean);
    }

    /// <summary>
    /// Сколько серий снимка известно профилям. Ноль означает, что снимок собран под другие имена серий:
    /// приора по нему не получит ни одна модель, и это тише пустого снимка
    /// </summary>
    public static int KnownSeries(BenchmarkSnapshot snapshot) => Profiles.Keys.Count(snapshot.Entries.ContainsKey);

    /// <summary>
    /// Уровень поля: средняя доля качества по всем строкам серий, известных профилям. С него стартует
    /// модель, которой в рейтингах нет (<see cref="Uniform(double)"/>); <c>null</c>, если известных серий нет
    /// </summary>
    /// <remarks>
    /// Качество в серии это доля между худшим и лучшим (<see cref="BenchmarkSnapshot.Quality"/>), и даже у
    /// сильнейших моделей в среднем по сериям оно около половины: лидер в каждой серии свой. Балл
    /// возможностей каталога лежит на другой шкале, около 0,8 у любой современной модели, и подставленный
    /// в <see cref="Uniform(double)"/> напрямую он ставил незнакомую модель выше всех оцененных на любой задаче.
    /// Если снимок несет среднюю полной серии (<see cref="SeriesBounds.Mean"/>), берется она: строки
    /// обрезанного снимка это верх серии, и средняя по ним завышала бы уровень поля.
    /// </remarks>
    public static double? FieldQuality(BenchmarkSnapshot snapshot)
    {
        (double Share, int Count)[] series = [.. Profiles.Keys
            .Select(snapshot.MeanShare)
            .Where(share => share is not null)
            .Select(share => share!.Value)];

        int rows = series.Sum(item => item.Count);

        return rows == 0 ? null : series.Sum(item => item.Share * item.Count) / rows;
    }

    /// <summary>
    /// Начальный вектор модели без рейтингов: качество <paramref name="quality"/> одинаково во всех сериях
    /// </summary>
    /// <remarks>
    /// Строится той же подгонкой по тем же профилям задач, что <see cref="GetVector(BenchmarkSnapshot, string)"/>,
    /// и потому лежит на одной шкале с моделями из рейтингов. Вектор по одной опорной задаче сжимается
    /// подгонкой иначе, чем вектор по полусотне точек, и безрейтинговая модель со средним баллом обгоняла
    /// сильнейшие рейтинговые.
    /// <para>
    /// Качество задается на шкале серий: незнакомой модели подходит уровень поля
    /// (<see cref="FieldQuality"/>), а балл каталога годится лишь как множитель к нему, не вместо него.
    /// </para>
    /// <para>
    /// Подгонка линейна по качеству, поэтому считается один раз для единицы и умножается: безрейтинговых
    /// моделей в каталоге сотни, и подгонка на каждую задерживала бы первый ход после обновления рейтингов.
    /// Запомненный вектор привязан к среднему задач: сменилось оно, считаем заново. Точки весят так же,
    /// как в <see cref="GetVector(BenchmarkSnapshot, string, Vector?)"/>.
    /// </para>
    /// </remarks>
    /// <param name="quality">Качество от 0 до 1 на шкале серий снимка</param>
    public static Vector Uniform(double quality) => Uniform(quality, Settings.TaskMean);

    /// <summary>Как <see cref="Uniform(double)"/>, в пространстве среднего <paramref name="mean"/></summary>
    /// <param name="quality">Качество от 0 до 1 на шкале серий снимка</param>
    /// <param name="mean">Среднее задач; пусто, значит без вычитания</param>
    public static Vector Uniform(double quality, Vector? mean)
    {
        UniformUnit? unit = _uniform;

        if (unit is null || !ReferenceEquals(unit.Mean, mean))
        {
            (Vector Task, double Quality, double Weight)[] points =
                [.. Profiles.Values.SelectMany(tasks => tasks.Select(task => (task.FeatureVector, 1.0, 1.0 / tasks.Count)))];

            _uniform = unit = new UniformUnit(mean,
                QualityPrior.Fit([.. points.Select(point => (point.Task, point.Quality))], [.. points.Select(point => point.Weight)], mean));
        }

        return unit.Vector * quality;
    }

    /// <summary>Скорость модели по внешнему замеру, токенов в секунду; нет замера, значит <c>null</c></summary>
    public static double? TokensPerSecond(BenchmarkSnapshot snapshot, string openRouterId) =>
        snapshot.Value(Speed, openRouterId) is > 0 and var speed ? speed : null;

    /// <summary>
    /// Скорость модели, которой в серии скорости нет: нижняя граница серии. Серия во встроенном снимке
    /// обрезана до самых быстрых, и модель вне нее медленнее последней строки, но не в разы: прежнее
    /// умолчание 50 токенов в секунду делало модель на 160 втрое медленнее модели на 170. Серии нет,
    /// значит <c>null</c>.
    /// </summary>
    public static double? DefaultTokensPerSecond(BenchmarkSnapshot snapshot) =>
        snapshot.BoundsOf(Speed) is { Low: > 0 } bounds ? bounds.Low : null;

    /// <summary>
    /// Начальная поправка цены: задача обходится дороже прайса ответа на долю рассуждений. Среднее
    /// по отраслевым индексам, где модель есть; нет ни в одном, значит <c>null</c>.
    /// </summary>
    public static double? CostRatio(BenchmarkSnapshot snapshot, string openRouterId)
    {
        double[] shares =
        [
            .. snapshot.KeysStartingWith(ReasoningShare)
                .Select(key => snapshot.Value(key, openRouterId))
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

    // Точки подгонки: задача профиля, качество модели в серии и вес 1/(число профилей серии)
    private static IEnumerable<(Vector Task, double Quality, double Weight)> Points(BenchmarkSnapshot snapshot, string openRouterId)
    {
        foreach ((string key, IReadOnlyList<InputFeatures> tasks) in Profiles)
            if (snapshot.Quality(key, openRouterId) is { } quality)
                foreach (InputFeatures task in tasks)
                    yield return (task.FeatureVector, quality, 1.0 / tasks.Count);
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

    private sealed record UniformUnit(Vector? Mean, Vector Vector);
}
