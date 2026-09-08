using System.Globalization;
using System.Text.Json;
using FAI.Router.Enums;
using FAI.Router.RoutedElements;
using FAI.Router.Services;

namespace FAI.Router.Catalog;

/// <summary>
/// Каталог моделей: цены, размеры окна и возможности берутся у OpenRouter, а не вбиваются руками.
/// </summary>
/// <remarks>
/// Скорость поставщик не публикует, поэтому число токенов в секунду задает вызывающий. Это
/// единственное, что каталог не закрывает, и в метрике R скорость участвует наравне с ценой.
/// </remarks>
public static class ModelCatalog
{
    /// <summary>
    /// Открытый список моделей OpenRouter, ключ для него не нужен
    /// </summary>
    public const string OpenRouterModels = "https://openrouter.ai/api/v1/models";

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    /// <summary>
    /// Загружает каталог у поставщика
    /// </summary>
    /// <param name="client">Готовый клиент; не задан, значит создается свой</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public static async Task<IReadOnlyList<ModelInfo>> FetchAsync(HttpClient? client = null, CancellationToken cancellationToken = default)
    {
        using HttpClient? own = client is null ? new HttpClient() : null;
        string json = await (client ?? own!).GetStringAsync(OpenRouterModels, cancellationToken).ConfigureAwait(false);

        return Parse(json);
    }

    /// <summary>
    /// Читает ранее сохраненный снимок каталога
    /// </summary>
    /// <param name="path">Путь к файлу снимка</param>
    public static IReadOnlyList<ModelInfo> Load(string path) =>
        JsonSerializer.Deserialize<ModelInfo[]>(File.ReadAllText(path)) ?? [];

    /// <summary>
    /// Сохраняет снимок каталога, чтобы состав и цены не менялись между прогонами
    /// </summary>
    /// <param name="path">Путь к файлу снимка</param>
    /// <param name="models">Что сохранять</param>
    public static void Save(string path, IEnumerable<ModelInfo> models) =>
        File.WriteAllText(path, JsonSerializer.Serialize(models, JsonOptions));

    /// <summary>
    /// Делает кандидата на исполнение из сведений каталога
    /// </summary>
    /// <param name="model">Сведения о модели</param>
    /// <param name="tokensPerSecond">Скорость, которой в каталоге нет</param>
    public static BaseRoutedElement CreateElement(ModelInfo model, double tokensPerSecond = 50) =>
        new()
        {
            Name = model.Id,
            DPMTInp = model.DollarsPerMillionInput,
            DPMTOutp = model.DollarsPerMillionOutput,
            TPS = tokensPerSecond,
            Capabilities = model.Capabilities,

            // ContextLimit измеряется в символах ответа, а поставщик считает в токенах
            ContextLimit = (int)(model.MaxAnswerTokens * InputFeaturesService.EST_SYMBOL_PER_TOKEN)
        };

    /// <summary>
    /// Разбирает ответ каталога
    /// </summary>
    /// <param name="json">Тело ответа</param>
    public static IReadOnlyList<ModelInfo> Parse(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        List<ModelInfo> models = [];

        foreach (JsonElement item in document.RootElement.GetProperty("data").EnumerateArray())
        {
            JsonElement pricing = item.GetProperty("pricing");
            JsonElement architecture = item.GetProperty("architecture");

            models.Add(new ModelInfo(
                item.GetProperty("id").GetString() ?? "",
                item.TryGetProperty("name", out JsonElement title) ? title.GetString() ?? "" : "",
                PerMillion(pricing, "prompt"),
                PerMillion(pricing, "completion"),
                Number(item, "context_length"),
                MaxAnswerTokens(item),
                CapabilitiesOf(item, architecture),
                Benchmark(item, "intelligence_index"),
                Benchmark(item, "coding_index"),
                Benchmark(item, "agentic_index")));
        }

        return models;
    }

    // Цены поставщик отдает строками и за один токен
    private static double PerMillion(JsonElement pricing, string field) =>
        pricing.TryGetProperty(field, out JsonElement value)
        && double.TryParse(value.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out double perToken)
            ? perToken * 1e6
            : 0;

    private static Capability CapabilitiesOf(JsonElement item, JsonElement architecture)
    {
        // Писать код и формулы умеет любая текстовая модель, это не отличительная черта.
        // Поиск в сети из каталога не выводится: он подключается отдельным дополнением.
        Capability capabilities = Capability.Code | Capability.Formulas;

        if (architecture.TryGetProperty("input_modalities", out JsonElement modalities)
            && modalities.EnumerateArray().Any(mode => mode.GetString() == "image"))
            capabilities |= Capability.Vision;

        if (item.TryGetProperty("supported_parameters", out JsonElement parameters)
            && parameters.EnumerateArray().Any(parameter => parameter.GetString() == "tools"))
            capabilities |= Capability.Tools;

        return capabilities;
    }

    private static int MaxAnswerTokens(JsonElement item) =>
        item.TryGetProperty("top_provider", out JsonElement provider)
            ? Number(provider, "max_completion_tokens")
            : 0;

    private static int Number(JsonElement item, string field) =>
        item.TryGetProperty(field, out JsonElement value) && value.ValueKind == JsonValueKind.Number
            ? value.GetInt32()
            : 0;

    private static double Benchmark(JsonElement item, string field) =>
        item.TryGetProperty("benchmarks", out JsonElement benchmarks)
        && benchmarks.TryGetProperty("artificial_analysis", out JsonElement analysis)
        && analysis.TryGetProperty(field, out JsonElement value)
        && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble()
            : 0;
}
