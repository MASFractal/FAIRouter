using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace FAI.Router.Catalog;

/// <summary>
/// Строка рейтинга: модель и ее оценка в серии. Голоса есть только у замеров предпочтений.
/// </summary>
public sealed record BenchmarkEntry(
    [property: JsonPropertyName("model_key")] string ModelKey,
    [property: JsonPropertyName("display_name")] string DisplayName,
    [property: JsonPropertyName("organization")] string Organization,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("votes")] int Votes = 0);

/// <summary>
/// Границы полной серии, записанные до того, как снимок обрезали до первых строк.
/// </summary>
/// <param name="Low">Худшая оценка в полной серии</param>
/// <param name="High">Лучшая оценка в полной серии</param>
/// <param name="Mean">Средняя оценка в полной серии; пусто, если не записана</param>
/// <param name="Count">Сколько строк было в полной серии</param>
public sealed record SeriesBounds(
    [property: JsonPropertyName("low")] double Low,
    [property: JsonPropertyName("high")] double High,
    [property: JsonPropertyName("mean")] double? Mean = null,
    [property: JsonPropertyName("count")] int Count = 0);

/// <summary>
/// Замеры по сериям на один момент времени: предпочтения людей (<c>pref:text/coding</c>) и
/// прогоны бенчмарков (<c>bench:index/legal</c>). Файл снимка общий с Python-версией, имена полей в JSON змеиные.
/// </summary>
/// <remarks>
/// Доля качества считается между худшим и лучшим в серии. Снимок в комплекте обрезан до первых строк
/// серии, и по обрезку размах предпочтений бывал 50 Elo: сильная модель в середине списка получала
/// долю 0,15. Поэтому снимок может нести границы полной серии (<see cref="Bounds"/>), их пишет
/// <see cref="Top"/> перед обрезкой; нет границ, значит доля по строкам, как раньше.
/// </remarks>
public sealed class BenchmarkSnapshot
{
    internal static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    /// <summary>
    /// Индексы серий по каноническому имени. Поиск зовут на каждую модель каталога и каждую серию при
    /// каждом выборе, и разбор имен всех строк серии на каждый вызов занимал секунды на выбор.
    /// </summary>
    private readonly ConditionalWeakTable<List<BenchmarkEntry>, SeriesIndex> _indexes = new();

    /// <summary>Ключи серий по префиксу, для поправки цены и подобных проходов по сериям</summary>
    private readonly ConcurrentDictionary<string, string[]> _prefixKeys = new();
    private Dictionary<string, List<BenchmarkEntry>>? _prefixSource;
    private int _prefixCount;

    [JsonPropertyName("fetched_at")]
    public string FetchedAt { get; set; } = "";

    [JsonPropertyName("entries")]
    public Dictionary<string, List<BenchmarkEntry>> Entries { get; set; } = [];

    /// <summary>
    /// Границы полных серий до обрезки; серии без записи считаются по своим строкам
    /// </summary>
    [JsonPropertyName("bounds")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, SeriesBounds>? Bounds { get; set; }

    /// <summary>Возраст снимка по <see cref="FetchedAt"/>; дата не разбирается, значит <c>null</c></summary>
    [JsonIgnore]
    public TimeSpan? Age => DateTimeOffset.TryParse(FetchedAt, out DateTimeOffset fetched) ? DateTimeOffset.UtcNow - fetched : null;

    /// <summary>Запись модели каталога в серии; нет, значит <c>null</c></summary>
    public BenchmarkEntry? Find(string key, string openRouterId) =>
        Entries.GetValueOrDefault(key) is { } rows ? IndexOf(key, rows).Find(openRouterId) : null;

    /// <summary>Сама оценка модели в серии (скорость, доля рассуждений); нет модели, значит <c>null</c></summary>
    public double? Value(string key, string openRouterId) => Find(key, openRouterId)?.Score;

    /// <summary>
    /// Качество модели в серии от 0 до 1: доля между худшим и лучшим в полной серии (по
    /// <see cref="Bounds"/>, без них по строкам). Лидер получает единицу. Абсолютный смысл прогнозу
    /// дает калибровка, поэтому шкала условна.
    /// </summary>
    public double? Quality(string key, string openRouterId)
    {
        if (Entries.GetValueOrDefault(key) is not { } rows) return null;
        SeriesIndex index = IndexOf(key, rows);

        return index.Find(openRouterId) is { } entry ? index.Share(entry.Score) : null;
    }

    /// <summary>Качество каждой строки серии на той же шкале, что <see cref="Quality"/>; пусто, если серии нет</summary>
    public IReadOnlyList<double> Shares(string key)
    {
        if (Entries.GetValueOrDefault(key) is not { } rows) return [];
        SeriesIndex index = IndexOf(key, rows);

        return [.. rows.Select(row => index.Share(row.Score))];
    }

    /// <summary>
    /// Средняя доля качества в серии и сколько строк за ней стоит: по средней полной серии, если она
    /// записана, иначе по строкам снимка. Нет серии, значит <c>null</c>.
    /// </summary>
    public (double Share, int Count)? MeanShare(string key)
    {
        if (Entries.GetValueOrDefault(key) is not { Count: > 0 } rows) return null;
        SeriesIndex index = IndexOf(key, rows);

        return Bounds?.GetValueOrDefault(key) is { Mean: { } mean } bounds
            ? (index.Share(mean), Math.Max(bounds.Count, rows.Count))
            : (rows.Average(row => index.Share(row.Score)), rows.Count);
    }

    /// <summary>Границы серии: записанные для полной серии, иначе по строкам; нет серии, значит <c>null</c></summary>
    public SeriesBounds? BoundsOf(string key) =>
        Bounds?.GetValueOrDefault(key)
        ?? (Entries.GetValueOrDefault(key) is { Count: > 0 } rows
            ? new SeriesBounds(rows.Min(row => row.Score), rows.Max(row => row.Score), rows.Average(row => row.Score), rows.Count)
            : null);

    /// <summary>Ключи серий с данным началом, в порядке снимка; список запоминается</summary>
    public IReadOnlyList<string> KeysStartingWith(string prefix)
    {
        if (!ReferenceEquals(_prefixSource, Entries) || _prefixCount != Entries.Count)
        {
            _prefixKeys.Clear();
            _prefixSource = Entries;
            _prefixCount = Entries.Count;
        }

        return _prefixKeys.GetOrAdd(prefix, start => [.. Entries.Keys.Where(key => key.StartsWith(start, StringComparison.Ordinal))]);
    }

    /// <summary>
    /// Снимок, обрезанный до первых count строк каждой серии: для тестов и работы без сети. Границы
    /// полных серий записываются до обрезки, поэтому доля качества по обрезку та же, что по полной серии.
    /// </summary>
    public BenchmarkSnapshot Top(int count) => new()
    {
        FetchedAt = FetchedAt,
        Entries = Entries.ToDictionary(pair => pair.Key, pair => pair.Value.Take(count).ToList()),
        Bounds = Entries.Keys
            .Select(key => (Key: key, Bounds: BoundsOf(key)))
            .Where(item => item.Bounds is not null)
            .ToDictionary(item => item.Key, item => item.Bounds!),
    };

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));

    public static BenchmarkSnapshot Load(string path) =>
        JsonSerializer.Deserialize<BenchmarkSnapshot>(File.ReadAllText(path)) ?? new BenchmarkSnapshot();

    /// <summary>
    /// Имя ресурса со снимком из комплекта сборки: тот же файл, что в пакете на Python
    /// </summary>
    public const string ResourceName = "FAI.Router.benchmark-snapshot.json";

    /// <summary>
    /// Снимок из комплекта сборки, чтобы кандидаты стартовали с прогноза по замерам, а не со
    /// случайного вектора, даже когда файла снимка рядом нет
    /// </summary>
    public static BenchmarkSnapshot LoadEmbedded()
    {
        using Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException($"В сборке нет ресурса {ResourceName}.");

        return JsonSerializer.Deserialize<BenchmarkSnapshot>(stream) ?? new BenchmarkSnapshot();
    }

    /// <summary>Индекс серии; список, выросший после построения, или новые границы индексируются заново.</summary>
    private SeriesIndex IndexOf(string key, List<BenchmarkEntry> rows)
    {
        SeriesBounds? bounds = Bounds?.GetValueOrDefault(key);

        if (_indexes.TryGetValue(rows, out SeriesIndex? index) && index.Count == rows.Count && ReferenceEquals(index.Bounds, bounds))
            return index;

        index = new SeriesIndex(rows, bounds);
        _indexes.AddOrUpdate(rows, index);
        return index;
    }

    /// <summary>
    /// Серия, разобранная один раз: лучшая запись на каждое каноническое имя по правилу
    /// <see cref="ModelNames.Find"/> (точное имя, затем семейство) и границы оценок для доли качества.
    /// </summary>
    private sealed class SeriesIndex
    {
        private readonly Dictionary<string, BenchmarkEntry> _exact = [];
        private readonly Dictionary<string, BenchmarkEntry> _family = [];
        private readonly double _low;
        private readonly double _high;

        public SeriesIndex(List<BenchmarkEntry> rows, SeriesBounds? bounds)
        {
            Count = rows.Count;
            Bounds = bounds;
            _low = bounds?.Low ?? (rows.Count == 0 ? 0 : rows.Min(row => row.Score));
            _high = bounds?.High ?? (rows.Count == 0 ? 0 : rows.Max(row => row.Score));

            foreach (BenchmarkEntry row in rows)
            {
                Add(_exact, row, relaxed: false);
                Add(_family, row, relaxed: true);
            }
        }

        public int Count { get; }

        public SeriesBounds? Bounds { get; }

        public BenchmarkEntry? Find(string openRouterId) =>
            _exact.GetValueOrDefault(ModelNames.Canonical(openRouterId))
            ?? _family.GetValueOrDefault(ModelNames.Canonical(openRouterId, relaxed: true));

        public double Share(double score) => _high <= _low ? 1.0 : Math.Clamp((score - _low) / (_high - _low), 0, 1);

        private static void Add(Dictionary<string, BenchmarkEntry> index, BenchmarkEntry row, bool relaxed)
        {
            foreach (string name in new[] { ModelNames.Canonical(row.DisplayName, relaxed), ModelNames.Canonical(row.ModelKey, relaxed) }.Distinct())
                if (!index.TryGetValue(name, out BenchmarkEntry? best) || ModelNames.Better(row, best))
                    index[name] = row;
        }
    }
}

/// <summary>
/// Сопоставление имен моделей каталога OpenRouter и внешних замеров.
/// </summary>
public static partial class ModelNames
{
    /// <summary>Сколько разобранных имен помнить: имена каталога и снимка конечны, предел только от утечки</summary>
    private const int MemoLimit = 20000;

    // Суффиксы, которыми источники различают одну и ту же модель: усилие размышления, режим,
    // дата выпуска, размер контекста. Одна модель каталога совпадает со всеми такими вариантами
    private static readonly HashSet<string> VariantTokens =
    [
        "text", "high", "medium", "low", "minimal", "xhigh", "instant",
        "search", "grounding", "preview", "exp", "reasoning", "non",
    ];

    // Суффиксы, которые у одних поставщиков означают режим той же модели (claude-opus-4-7-thinking,
    // «GLM-5.2 (max)»), а у других отдельный продукт (qwen3.8-max, kimi-k2-thinking, gpt-5.1-codex-max).
    // Точное имя их сохраняет, семейство срезает: сначала ищется точное совпадение, затем семейство.
    // Исключение одно: у Claude размышление это режим той же модели каталога, отдельного продукта нет
    private static readonly HashSet<string> ProductTokens = ["max", "thinking"];

    private const string ClaudeThinking = "thinking";

    private static readonly ConcurrentDictionary<(string Name, bool Relaxed), string> Memo = new();

    /// <summary>
    /// Имя модели без поставщика, вариантов усилия, режима, даты и контекста. «anthropic/claude-opus-4.7»,
    /// «claude-opus-4-7-high» и «claude-opus-4-7-20251101-high-32k» дают одно и то же, как и
    /// «anthropic/claude-haiku-4.5» с «claude-4-5-haiku-reasoning» (иной порядок слов в имени).
    /// </summary>
    /// <remarks>
    /// Хвосты max и thinking точное имя сохраняет: qwen3.8-max и qwen3.8, gpt-5.1-codex-max и
    /// gpt-5.1-codex это разные продукты, и прежде они склеивались в одно имя. Семейство
    /// (<paramref name="relaxed"/>) срезает и их: так модель каталога находит свою строку, когда в серии
    /// есть только вариант с режимом.
    /// </remarks>
    /// <param name="name">Имя из каталога или из серии</param>
    /// <param name="relaxed">Срезать ли и хвосты продукта (семейство вместо точного имени)</param>
    public static string Canonical(string name, bool relaxed = false)
    {
        if (Memo.TryGetValue((name, relaxed), out string? known))
            return known;

        if (Memo.Count > MemoLimit)
            Memo.Clear();

        return Memo[(name, relaxed)] = Parse(name, relaxed);
    }

    /// <summary>
    /// Запись для модели каталога: среди совпавших по точному каноническому имени берется та, у которой
    /// больше голосов, а при равных голосах самая короткая, то есть вариант без суффиксов. Точных нет,
    /// тогда так же среди совпавших по семейству.
    /// </summary>
    public static BenchmarkEntry? Find(string openRouterId, IEnumerable<BenchmarkEntry> entries)
    {
        BenchmarkEntry[] rows = [.. entries];

        return Best(rows, Canonical(openRouterId), relaxed: false) ?? Best(rows, Canonical(openRouterId, relaxed: true), relaxed: true);
    }

    /// <summary>Часть ключа серии из названия источника: «Finance/Investing» становится «finance-investing»</summary>
    public static string Slug(string name) => NonAlphanumeric().Replace(name.ToLowerInvariant(), "-").Trim('-');

    /// <summary>Запись лучше другой той же модели: больше голосов, при равных короче ключ; при полном равенстве остается прежняя</summary>
    internal static bool Better(BenchmarkEntry row, BenchmarkEntry than) =>
        row.Votes > than.Votes || (row.Votes == than.Votes && row.ModelKey.Length < than.ModelKey.Length);

    private static BenchmarkEntry? Best(BenchmarkEntry[] rows, string wanted, bool relaxed)
    {
        BenchmarkEntry? best = null;

        foreach (BenchmarkEntry row in rows)
            if ((Canonical(row.DisplayName, relaxed) == wanted || Canonical(row.ModelKey, relaxed) == wanted) && (best is null || Better(row, best)))
                best = row;

        return best;
    }

    private static string Parse(string name, bool relaxed)
    {
        string text = name.Trim().ToLowerInvariant();
        int slash = text.IndexOf('/');
        if (slash >= 0) text = text[(slash + 1)..];
        int colon = text.IndexOf(':');
        if (colon >= 0) text = text[..colon];

        List<string> tokens = [.. Separators().Replace(text, "-").Trim('-').Split('-')];

        bool claude = tokens[0] == "claude";

        while (tokens.Count > 1 && (IsVariant(tokens[^1], relaxed) || (claude && tokens[^1] == ClaudeThinking)))
            tokens.RemoveAt(tokens.Count - 1);

        // Дата вида 2024-05-13 после разбиения по дефису стала тремя числами
        while (tokens.Count > 3 && IsDigits(tokens[^3], 4) && IsDigits(tokens[^2], 2) && IsDigits(tokens[^1], 2))
            tokens.RemoveRange(tokens.Count - 3, 3);

        return ClaudeOldOrder().Replace(string.Join('-', tokens.Where(token => token.Length > 0)), "claude-$2-$1");
    }

    private static bool IsVariant(string token, bool relaxed) =>
        VariantTokens.Contains(token) || VariantTail().IsMatch(token) || (relaxed && ProductTokens.Contains(token));

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
