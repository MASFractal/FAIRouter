using System.Text.Json;
using System.Text.RegularExpressions;

namespace FAI.Router.Catalog;

/// <summary>
/// Категория арены: арена (text, code, search), имя категории из адреса и часть ключа рейтинга,
/// если она не совпадает с именем.
/// </summary>
public readonly record struct ArenaCategory(string Arena, string Name, string? Slug = null)
{
    /// <summary>Ключ серии в снимке</summary>
    public string Key => $"arena:{Arena}/{Name}";

    /// <summary>Адрес страницы</summary>
    public string Url => $"https://arena.ai/leaderboard/{Arena}/{Name}";

    /// <summary>Что обязано встретиться в ключе рейтинга, отданного страницей</summary>
    public string ExpectedKeyPart => Slug ?? Name.Replace('-', '_');
}

/// <summary>
/// Рейтинги арены (arena.ai): реестр категорий и разбор страниц.
/// </summary>
/// <remarks>
/// Рейтинг фактологии отдельной страницей не отдается: переключатель веса фактологии работает в
/// браузере, адрес <c>text/overall-factuality</c> возвращает общий рейтинг без контроля стиля.
/// Поэтому фактология берется у Artificial Analysis (<see cref="AnalysisLeaderboard"/>).
/// </remarks>
public static partial class ArenaLeaderboard
{
    /// <summary>
    /// Все 29 текстовых категорий арены, кроме служебной exclude-ties. Список повторяет реестр
    /// страницы leaderboard/text.
    /// </summary>
    private static readonly string[] TextCategories =
    [
        "overall", "expert",
        "industry-software-and-it-services", "industry-writing-and-literature-and-language",
        "industry-life-and-physical-and-social-science", "industry-entertainment-and-sports-and-media",
        "industry-business-and-management-and-financial-operations", "industry-mathematical",
        "industry-legal-and-government", "industry-medicine-and-healthcare",
        "math", "instruction-following", "multi-turn", "creative-writing", "coding",
        "hard-prompts", "hard-prompts-english", "longer-query",
        "english", "non-english", "chinese", "french", "german", "spanish", "russian", "japanese",
        "korean", "polish",
    ];

    /// <summary>Текстовые категории, пять категорий арены кода и поисковая арена</summary>
    public static readonly IReadOnlyList<ArenaCategory> Categories =
    [
        .. TextCategories.Select(name => new ArenaCategory("text", name)),
        new("code", "frontend"),
        new("code", "fullstack"),
        new("code", "html"),
        new("code", "react"),
        new("code", "brand-marketing"),
        new("search", "overall", "search-overall"),
    ];

    /// <summary>
    /// Ключ рейтинга и его записи со страницы арены. Страница без рейтинга дает исключение.
    /// </summary>
    /// <param name="html">Тело страницы</param>
    public static (string Key, IReadOnlyList<BenchmarkEntry> Entries) Parse(string html)
    {
        string text = Rsc.Payload(html);
        Match found = EntriesStart().Match(text);

        if (!found.Success)
            throw new FormatException("На странице нет рейтинга: адрес категории не тот или разметка изменилась.");

        using JsonDocument document = JsonDocument.Parse(Rsc.ArrayAt(text, found.Index + found.Length));
        List<BenchmarkEntry> rows = [];

        foreach (JsonElement row in document.RootElement.EnumerateArray())
            rows.Add(new BenchmarkEntry(
                Text(row, "modelKey"), Text(row, "modelDisplayName"), Text(row, "modelOrganization"),
                row.TryGetProperty("rating", out JsonElement rating) ? rating.GetDouble() : 0,
                row.TryGetProperty("votes", out JsonElement votes) ? votes.GetInt32() : 0));

        return (found.Groups[1].Value, rows);
    }

    /// <summary>
    /// Рейтинг категории с сайта. Ключ рейтинга сверяется с категорией: страница неизвестного
    /// адреса отдает общий рейтинг, и без проверки категория молча подменилась бы общей.
    /// </summary>
    public static async Task<IReadOnlyList<BenchmarkEntry>> FetchAsync(
        ArenaCategory category, HttpClient? client = null, CancellationToken cancellationToken = default)
    {
        (string key, IReadOnlyList<BenchmarkEntry> rows) =
            Parse(await Rsc.GetAsync(category.Url, client, cancellationToken).ConfigureAwait(false));

        return key.Contains(category.ExpectedKeyPart)
            ? rows
            : throw new FormatException($"Страница {category.Url} отдала рейтинг {key}, а не {category.Name}.");
    }

    /// <summary>Серии по всем категориям реестра (или по заданным)</summary>
    public static async Task<Dictionary<string, List<BenchmarkEntry>>> FetchAllAsync(
        HttpClient? client = null, IEnumerable<ArenaCategory>? categories = null, CancellationToken cancellationToken = default)
    {
        Dictionary<string, List<BenchmarkEntry>> series = [];

        foreach (ArenaCategory category in categories ?? Categories)
            series[category.Key] = [.. await FetchAsync(category, client, cancellationToken).ConfigureAwait(false)];

        return series;
    }

    private static string Text(JsonElement row, string name) =>
        row.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString()! : "";

    [GeneratedRegex("""leaderboards/([a-z0-9_\-]+)/leaderboard-snapshots/latest","entries":""")]
    private static partial Regex EntriesStart();
}
