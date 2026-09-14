using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace FAI.Router.Catalog;

/// <summary>
/// Строка рейтинга: модель и ее оценка в серии. Голоса есть только у арены.
/// </summary>
public sealed record BenchmarkEntry(
    [property: JsonPropertyName("model_key")] string ModelKey,
    [property: JsonPropertyName("display_name")] string DisplayName,
    [property: JsonPropertyName("organization")] string Organization,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("votes")] int Votes = 0);

/// <summary>
/// Рейтинги по сериям на один момент времени: арена (<c>arena:text/coding</c>) и Artificial
/// Analysis (<c>aa:index/legal</c>). Файл снимка общий с Python-версией, имена полей в JSON змеиные.
/// </summary>
public sealed class BenchmarkSnapshot
{
    internal static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    [JsonPropertyName("fetched_at")]
    public string FetchedAt { get; set; } = "";

    [JsonPropertyName("entries")]
    public Dictionary<string, List<BenchmarkEntry>> Entries { get; set; } = [];

    /// <summary>Запись модели каталога в серии; нет, значит <c>null</c></summary>
    public BenchmarkEntry? Find(string key, string openRouterId) =>
        ModelNames.Find(openRouterId, Entries.GetValueOrDefault(key) ?? []);

    /// <summary>Сама оценка модели в серии (скорость, доля рассуждений); нет модели, значит <c>null</c></summary>
    public double? Value(string key, string openRouterId) => Find(key, openRouterId)?.Score;

    /// <summary>
    /// Качество модели в серии от 0 до 1: доля между худшим и лучшим. Лидер получает единицу.
    /// Абсолютный смысл прогнозу дает калибровка, поэтому шкала условна.
    /// </summary>
    public double? Quality(string key, string openRouterId)
    {
        List<BenchmarkEntry> rows = Entries.GetValueOrDefault(key) ?? [];
        BenchmarkEntry? entry = ModelNames.Find(openRouterId, rows);

        if (entry is null)
            return null;

        double low = rows.Min(row => row.Score);
        double high = rows.Max(row => row.Score);

        return high <= low ? 1.0 : (entry.Score - low) / (high - low);
    }

    /// <summary>Снимок, обрезанный до первых count строк каждой серии: для тестов и работы без сети</summary>
    public BenchmarkSnapshot Top(int count) => new()
    {
        FetchedAt = FetchedAt,
        Entries = Entries.ToDictionary(pair => pair.Key, pair => pair.Value.Take(count).ToList()),
    };

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));

    public static BenchmarkSnapshot Load(string path) =>
        JsonSerializer.Deserialize<BenchmarkSnapshot>(File.ReadAllText(path)) ?? new BenchmarkSnapshot();

    /// <summary>Снимок по обоим источникам</summary>
    public static async Task<BenchmarkSnapshot> FetchAllAsync(HttpClient? client = null, CancellationToken cancellationToken = default)
    {
        BenchmarkSnapshot snapshot = new() { FetchedAt = DateTimeOffset.UtcNow.ToString("O") };

        foreach ((string key, List<BenchmarkEntry> rows) in await ArenaLeaderboard.FetchAllAsync(client, cancellationToken: cancellationToken).ConfigureAwait(false))
            snapshot.Entries[key] = rows;

        foreach ((string key, List<BenchmarkEntry> rows) in await AnalysisLeaderboard.FetchAllAsync(client, cancellationToken).ConfigureAwait(false))
            snapshot.Entries[key] = rows;

        return snapshot;
    }
}

/// <summary>
/// Сопоставление имен моделей каталога OpenRouter, арены и Artificial Analysis.
/// </summary>
public static partial class ModelNames
{
    // Суффиксы имен, которыми источники различают одну и ту же модель: усилие размышления, режим,
    // дата выпуска, размер контекста. Одна модель каталога совпадает со всеми такими вариантами
    private static readonly HashSet<string> VariantTokens =
    [
        "text", "high", "max", "medium", "low", "minimal", "xhigh", "instant", "thinking",
        "search", "grounding", "preview", "exp", "reasoning", "non",
    ];

    /// <summary>
    /// Имя модели без поставщика, вариантов усилия, режима, даты и контекста. «anthropic/claude-opus-4.7»,
    /// «claude-opus-4-7-high» и «claude-opus-4-7-20251101-high-32k» дают одно и то же, как и
    /// «anthropic/claude-haiku-4.5» с «claude-4-5-haiku-reasoning» (старый порядок имени у AA).
    /// </summary>
    public static string Canonical(string name)
    {
        string text = name.Trim().ToLowerInvariant();
        int slash = text.IndexOf('/');
        if (slash >= 0) text = text[(slash + 1)..];
        int colon = text.IndexOf(':');
        if (colon >= 0) text = text[..colon];

        List<string> tokens = [.. Separators().Replace(text, "-").Trim('-').Split('-')];

        while (tokens.Count > 1 && (VariantTokens.Contains(tokens[^1]) || VariantTail().IsMatch(tokens[^1])))
            tokens.RemoveAt(tokens.Count - 1);

        // Дата вида 2024-05-13 после разбиения по дефису стала тремя числами
        while (tokens.Count > 3 && IsDigits(tokens[^3], 4) && IsDigits(tokens[^2], 2) && IsDigits(tokens[^1], 2))
            tokens.RemoveRange(tokens.Count - 3, 3);

        return ClaudeOldOrder().Replace(string.Join('-', tokens.Where(token => token.Length > 0)), "claude-$2-$1");
    }

    /// <summary>
    /// Запись для модели каталога: среди совпавших по каноническому имени берется та, у которой
    /// больше голосов, а при равных голосах самая короткая, то есть вариант без суффиксов.
    /// </summary>
    public static BenchmarkEntry? Find(string openRouterId, IEnumerable<BenchmarkEntry> entries)
    {
        string wanted = Canonical(openRouterId);

        return entries
            .Where(row => Canonical(row.DisplayName) == wanted || Canonical(row.ModelKey) == wanted)
            .OrderByDescending(row => row.Votes)
            .ThenBy(row => row.ModelKey.Length)
            .FirstOrDefault();
    }

    /// <summary>Часть ключа серии из названия источника: «Finance/Investing» становится «finance-investing»</summary>
    public static string Slug(string name) => NonAlphanumeric().Replace(name.ToLowerInvariant(), "-").Trim('-');

    private static bool IsDigits(string token, int length) => token.Length == length && token.All(char.IsAsciiDigit);

    [GeneratedRegex(@"[().,\s]+")]
    private static partial Regex Separators();

    [GeneratedRegex(@"^(\d{8}|\d{4}-\d{2}-\d{2}|\d{1,4}k|beta\d*)$")]
    private static partial Regex VariantTail();

    [GeneratedRegex(@"^claude-(\d+(?:-\d+)?)-(haiku|sonnet|opus)(?=-|$)")]
    private static partial Regex ClaudeOldOrder();

    [GeneratedRegex("[^a-z0-9]+")]
    private static partial Regex NonAlphanumeric();
}
