using System.Text.Json;
using System.Text.RegularExpressions;

namespace FAI.Router.Catalog;

/// <summary>
/// Рейтинги Artificial Analysis (artificialanalysis.ai): отраслевые индексы, фактология, бизнес-
/// функции агентной работы, документная работа по отраслям, скорость и цена задачи.
/// </summary>
/// <remarks>
/// Официальный API требует ключа, а страницы отдают массив моделей со всеми оценками внутри кусков
/// RSC, как и арена. Берутся восемь страниц: шесть индексов, страница AutomationBench (передовые
/// модели со всеми разбивками) и общая таблица (все модели, основные оценки, скорость). Серии, где
/// меньше значит лучше, не берутся: вместо доли выдумки берется доля ответов без выдумки.
/// </remarks>
public static partial class AnalysisLeaderboard
{
    private const string BaseUrl = "https://artificialanalysis.ai/";

    /// <summary>Отраслевые индексы: адрес models/capabilities/&lt;индекс&gt;</summary>
    public static readonly IReadOnlyList<string> Indices =
        ["finance-and-accounting", "strategy-and-ops", "legal", "healthcare-and-medical", "engineering", "economics"];

    public const string EvaluationPath = "evaluations/automationbench-aa";
    public const string LeaderboardPath = "leaderboards/models";

    /// <summary>Серии общей таблицы: ключ снимка и поле модели</summary>
    public static readonly IReadOnlyList<(string Key, string Field)> TableSeriesFields =
    [
        ("aa:intelligence", "intelligenceIndex"),
        ("aa:non-hallucination", "omniscienceNonHallucination"),
        ("aa:ifbench", "ifbench"),
        ("aa:lcr", "lcr"),
        ("aa:tau2", "tau2"),
        ("aa:hle", "hle"),
        ("aa:gpqa", "gpqa"),
        ("aa:gdpval", "gdpvalNormalized"),
        ("aa:terminalbench-hard", "terminalbenchHard"),
        ("aa:scicode", "scicode"),
        ("aa:speed", "medianOutputTokensPerSecond"),
    ];

    /// <summary>
    /// Массив моделей страницы. На странице бывает несколько массивов (меню, выборка для графика),
    /// берется тот, у записей которого больше всего полей.
    /// </summary>
    public static IReadOnlyList<JsonElement> ModelsOf(string html)
    {
        string text = Rsc.Payload(html);
        JsonElement[] best = [];
        int bestFields = -1;

        foreach (Match match in ModelArray().Matches(text))
        {
            using JsonDocument document = JsonDocument.Parse(Rsc.ArrayAt(text, match.Index + match.Length - 1));
            JsonElement[] rows = [.. document.RootElement.EnumerateArray().Select(row => row.Clone())];

            if (rows.Length == 0 || rows[0].ValueKind != JsonValueKind.Object)
                continue;

            int fields = rows[0].EnumerateObject().Count();

            if (fields > bestFields)
            {
                best = rows;
                bestFields = fields;
            }
        }

        return best.Length > 0
            ? best
            : throw new FormatException("На странице нет массива моделей: адрес не тот или разметка изменилась.");
    }

    /// <summary>
    /// Индекс, его части, цена задачи и доля рассуждений в ней. Доля рассуждений дает начальную
    /// поправку цены: прайс не знает, сколько модель потратит на размышление.
    /// </summary>
    public static Dictionary<string, List<BenchmarkEntry>> IndexSeries(string index, IReadOnlyList<JsonElement> models)
    {
        Dictionary<string, List<BenchmarkEntry>> series = [];

        foreach (JsonElement model in models)
        {
            Add(series, $"aa:index/{index}", model, Number(model, "weightedIndex"));

            // Части индекса названы ключами JSON в camelCase: nonHallucination становится non-hallucination
            foreach (JsonProperty part in Properties(model, "subScores"))
                Add(series, $"aa:index/{index}/{ModelNames.Slug(CamelBoundary().Replace(part.Name, "$1-$2"))}", model, AsNumber(part.Value));

            JsonElement cost = Child(model, "costPerTask");
            double? total = Number(cost, "total");
            double? reasoning = Number(cost, "reasoning");

            Add(series, $"aa:cost-per-task/{index}", model, total);

            if (total > 0 && reasoning is not null)
                Add(series, $"aa:reasoning-share/{index}", model, reasoning / total);
        }

        return series;
    }

    /// <summary>
    /// Разбивки передовых моделей: фактология по областям знаний и языкам программирования,
    /// бизнес-функции и сервисы AutomationBench, отрасли GDP.pdf, аналитика и оформление
    /// Briefcase, практика права Harvey.
    /// </summary>
    public static Dictionary<string, List<BenchmarkEntry>> EvaluationSeries(IReadOnlyList<JsonElement> models)
    {
        Dictionary<string, List<BenchmarkEntry>> series = [];

        foreach (JsonElement model in models)
        {
            JsonElement omniscience = Child(model, "omniscienceBreakdown");

            foreach (JsonProperty item in Properties(omniscience, "byDomain"))
                Add(series, $"aa:omniscience/{ModelNames.Slug(item.Name)}", model, AsNumber(item.Value));

            foreach (JsonProperty item in Properties(omniscience, "bySweLanguage"))
                Add(series, $"aa:swe-language/{ModelNames.Slug(item.Name)}", model, AsNumber(item.Value));

            JsonElement automation = Child(model, "automationBenchBreakdown");

            foreach (JsonProperty item in Properties(automation, "byDomain"))
                Add(series, $"aa:automation/{ModelNames.Slug(item.Name)}", model, Number(item.Value, "completion"));

            foreach (JsonProperty item in Properties(automation, "byApp"))
                Add(series, $"aa:automation-app/{ModelNames.Slug(item.Name)}", model, Number(item.Value, "completion"));

            foreach (JsonProperty item in Properties(Child(model, "gdpPdfBreakdown"), "byDomain"))
                Add(series, $"aa:gdp-pdf/{ModelNames.Slug(item.Name)}", model, AsNumber(item.Value));

            JsonElement briefcase = Child(model, "briefcaseBreakdown");
            Add(series, "aa:briefcase/analytical", model, Number(Child(briefcase, "analyticalQuality"), "elo"));
            Add(series, "aa:briefcase/presentation", model, Number(Child(briefcase, "presentation"), "elo"));
            Add(series, "aa:harvey", model, Number(model, "harveyLab"));
        }

        return series;
    }

    /// <summary>Основные оценки, скорость и цены по общей таблице</summary>
    public static Dictionary<string, List<BenchmarkEntry>> TableSeries(IReadOnlyList<JsonElement> models)
    {
        Dictionary<string, List<BenchmarkEntry>> series = [];

        foreach (JsonElement model in models)
            foreach ((string key, string field) in TableSeriesFields)
                Add(series, key, model, Number(model, field));

        return series;
    }

    /// <summary>Все серии: восемь страниц. Внутри серии порядок по убыванию оценки, как у арены</summary>
    public static async Task<Dictionary<string, List<BenchmarkEntry>>> FetchAllAsync(
        HttpClient? client = null, CancellationToken cancellationToken = default)
    {
        Dictionary<string, List<BenchmarkEntry>> series = [];

        foreach (string index in Indices)
            Merge(series, IndexSeries(index, ModelsOf(await Rsc.GetAsync(BaseUrl + "models/capabilities/" + index, client, cancellationToken).ConfigureAwait(false))));

        Merge(series, EvaluationSeries(ModelsOf(await Rsc.GetAsync(BaseUrl + EvaluationPath, client, cancellationToken).ConfigureAwait(false))));
        Merge(series, TableSeries(ModelsOf(await Rsc.GetAsync(BaseUrl + LeaderboardPath, client, cancellationToken).ConfigureAwait(false))));

        return series.ToDictionary(pair => pair.Key, pair => pair.Value.OrderByDescending(row => row.Score).ToList());
    }

    private static void Merge(Dictionary<string, List<BenchmarkEntry>> target, Dictionary<string, List<BenchmarkEntry>> source)
    {
        foreach ((string key, List<BenchmarkEntry> rows) in source)
            target[key] = rows;
    }

    private static void Add(Dictionary<string, List<BenchmarkEntry>> series, string key, JsonElement model, double? value)
    {
        if (value is null)
            return;

        JsonElement creator = Child(model, "creator");
        string name = Text(model, "name") is { Length: > 0 } full ? full : Text(model, "shortName");
        string organization = Text(creator, "name") is { Length: > 0 } owner ? owner : Text(model, "modelCreatorName");

        if (!series.TryGetValue(key, out List<BenchmarkEntry>? rows))
            series[key] = rows = [];

        rows.Add(new BenchmarkEntry(Text(model, "slug"), name, organization, value.Value));
    }

    private static JsonElement Child(JsonElement element, string name) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out JsonElement child) ? child : default;

    private static IEnumerable<JsonProperty> Properties(JsonElement element, string name)
    {
        JsonElement child = Child(element, name);

        return child.ValueKind == JsonValueKind.Object ? child.EnumerateObject() : [];
    }

    private static double? Number(JsonElement element, string name) => AsNumber(Child(element, name));

    private static double? AsNumber(JsonElement element) =>
        element.ValueKind == JsonValueKind.Number ? element.GetDouble() : null;

    private static string Text(JsonElement element, string name)
    {
        JsonElement child = Child(element, name);

        return child.ValueKind == JsonValueKind.String ? child.GetString()! : "";
    }

    [GeneratedRegex("\"(?:initialModels|models)\":\\[")]
    private static partial Regex ModelArray();

    [GeneratedRegex("([a-z0-9])([A-Z])")]
    private static partial Regex CamelBoundary();
}
