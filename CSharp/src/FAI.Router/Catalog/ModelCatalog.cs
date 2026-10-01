using System.Globalization;
using System.Reflection;
using System.Net.Http.Headers;
using System.Text.Json;
using FAI.Router.Enums;
using FAI.Router.RoutedElements;
using FAI.Router.Services;
using FAI.Router.Training;
using AI.DataStructs.Algebraic;

namespace FAI.Router.Catalog;

/// <summary>
/// Каталог моделей: цены, размеры окна и возможности берутся у поставщика, FractalRouter или OpenRouter,
/// а не вбиваются руками.
/// </summary>
/// <remarks>
/// Скорость поставщики не публикуют, поэтому число токенов в секунду задает вызывающий. Это
/// единственное, что каталог не закрывает, и в метрике R скорость участвует наравне с ценой.
/// Цены хранятся в валюте поставщика за миллион токенов: у OpenRouter доллары, у FractalRouter рубли.
/// Для выбора важны только отношения цен кандидатов между собой, поэтому валюта роли не играет, лишь бы
/// все кандидаты одного роутера были из одного каталога.
/// </remarks>
public static class ModelCatalog
{
    /// <summary>
    /// Открытый список моделей OpenRouter, ключ для него не нужен
    /// </summary>
    public const string OpenRouterModels = "https://openrouter.ai/api/v1/models";

    /// <summary>
    /// Список моделей FractalRouter, отдается только по ключу
    /// </summary>
    public const string FractalRouterModels = "https://api.fractalrouter.ru/v1/models";

    /// <summary>
    /// Скорость модели, о которой ничего не известно, токенов в секунду: обычное значение для
    /// нынешнего поколения моделей
    /// </summary>
    public const double DefaultTokensPerSecond = 50;

    /// <summary>
    /// Именованный набор вместо списка идентификаторов: весь каталог поставщика
    /// </summary>
    public const string All = "all";

    /// <summary>
    /// Именованный набор вместо списка идентификаторов: популярные модели из комплекта сборки,
    /// которые есть в каталоге поставщика. Выбор по умолчанию, когда модели не названы.
    /// </summary>
    public const string Popular = "popular";

    /// <summary>
    /// Ресурс со списком популярных моделей: тот же файл, что в пакете на Python
    /// </summary>
    public const string PopularResourceName = "FAI.Router.popular_models.json";

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    /// <summary>
    /// Загружает каталог OpenRouter, ключ не нужен
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
    /// Загружает каталог FractalRouter: цены в рублях за миллион токенов
    /// </summary>
    /// <param name="apiKey">Ключ FractalRouter, без него каталог не отдается</param>
    /// <param name="client">Готовый клиент; не задан, значит создается свой</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public static async Task<IReadOnlyList<ModelInfo>> FetchFractalRouterAsync(string apiKey, HttpClient? client = null, CancellationToken cancellationToken = default)
    {
        using HttpClient? own = client is null ? new HttpClient() : null;
        using HttpRequestMessage request = new(HttpMethod.Get, FractalRouterModels);
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        using HttpResponseMessage response = await (client ?? own!).SendAsync(request, cancellationToken).ConfigureAwait(false);
        string json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"Каталог FractalRouter не отдан: {(int)response.StatusCode} {response.ReasonPhrase}: {json}");

        return ParseFractalRouter(json);
    }

    /// <summary>
    /// Идентификаторы популярных моделей из комплекта: модели с внешними рейтингами в снимке замеров и
    /// недорогие рабочие лошадки
    /// </summary>
    public static IReadOnlyList<string> PopularModels()
    {
        using Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(PopularResourceName)
            ?? throw new InvalidOperationException($"В сборке нет ресурса {PopularResourceName}.");
        using JsonDocument document = JsonDocument.Parse(stream);

        return [.. document.RootElement.GetProperty("models").EnumerateArray()
            .Select(item => item.GetProperty("id").GetString() ?? "")
            .Where(id => id.Length > 0)];
    }

    /// <summary>
    /// Список идентификаторов по тому, что передали: один элемент «all» дает весь каталог поставщика,
    /// «popular» дает популярные из комплекта, которые есть в каталоге, иной список берется как есть.
    /// Strict говорит, обязан ли каждый идентификатор найтись: у именованного набора отсутствующие
    /// модели молча пропускаются, у списка их отсутствие это ошибка.
    /// </summary>
    /// <param name="modelIds">Список идентификаторов либо имя набора одним элементом; пусто, значит popular</param>
    /// <param name="known">Каталог поставщика по идентификаторам</param>
    public static (IReadOnlyList<string> Ids, bool Strict) Select(IEnumerable<string>? modelIds, IReadOnlyDictionary<string, ModelInfo> known)
    {
        string[] ids = modelIds is null ? [Popular] : [.. modelIds];

        if (ids.Length == 1 && string.Equals(ids[0], All, StringComparison.OrdinalIgnoreCase))
            return ([.. known.Values.Where(model => model.DollarsPerMillionInput >= 0 && model.DollarsPerMillionOutput >= 0).Select(model => model.Id)], false);

        if (ids.Length == 1 && string.Equals(ids[0], Popular, StringComparison.OrdinalIgnoreCase))
        {
            string[] found = [.. PopularModels().Where(known.ContainsKey)];

            if (found.Length == 0)
                throw new InvalidOperationException("Ни одной популярной модели в каталоге поставщика нет: назовите модели списком.");

            return (found, false);
        }

        return (ids, true);
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
    /// <remarks>
    /// Со снимком рейтингов кандидат стартует не со случайного вектора, а с прогноза по сериям внешних замеров
    /// (<see cref="BenchmarkPrior"/>). Оттуда же берутся скорость, если ее не
    /// назвали, и поправка цены на рассуждения. Модели, которой в рейтингах нет, снимок не касается.
    /// </remarks>
    /// <param name="model">Сведения о модели</param>
    /// <param name="tokensPerSecond">Скорость, которой в каталоге нет; не задана, тогда из рейтингов или 50</param>
    /// <param name="benchmarks">Снимок рейтингов для начального вектора; пусто, значит вектор случайный</param>
    public static BaseRoutedElement CreateElement(ModelInfo model, double? tokensPerSecond = null, BenchmarkSnapshot? benchmarks = null)
    {
        BaseRoutedElement element = new()
        {
            Name = model.Id,
            DPMTInp = model.DollarsPerMillionInput,
            DPMTOutp = model.DollarsPerMillionOutput,
            TPS = tokensPerSecond ?? (benchmarks is null ? null : BenchmarkPrior.TokensPerSecond(benchmarks, model.Id)) ?? DefaultTokensPerSecond,
            Capabilities = model.Capabilities,

            // ContextLimit измеряется в символах ответа, а поставщик считает в токенах
            ContextLimit = (int)(model.MaxAnswerTokens * InputFeaturesService.EST_SYMBOL_PER_TOKEN)
        };

        if (benchmarks is null)
            return element;

        element.IdealMatchVector = BenchmarkPrior.GetVector(benchmarks, model.Id) ?? element.IdealMatchVector;
        element.CostRatio = BenchmarkPrior.CostRatio(benchmarks, model.Id) ?? element.CostRatio;

        return element;
    }

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
                CapabilitiesOf(item, architecture)));
        }

        return models;
    }

    /// <summary>
    /// Разбирает ответ каталога FractalRouter: цены уже за миллион токенов, в рублях. Предел ответа
    /// каталог не сообщает, поэтому ограничения по объему у кандидата нет.
    /// </summary>
    /// <param name="json">Тело ответа</param>
    public static IReadOnlyList<ModelInfo> ParseFractalRouter(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        JsonElement items = root.ValueKind == JsonValueKind.Array ? root : root.GetProperty("data");
        List<ModelInfo> models = [];

        foreach (JsonElement item in items.EnumerateArray())
        {
            JsonElement pricing = item.TryGetProperty("pricing", out JsonElement found) ? found : default;

            // Умение вызывать инструменты каталог не сообщает; считаем, что умеют все, иначе запрос
            // с required: Capability.Tools не нашел бы ни одного кандидата
            Capability capabilities = Capability.Code | Capability.Formulas | Capability.Tools;

            if (item.TryGetProperty("modalities", out JsonElement modalities)
                && modalities.ValueKind == JsonValueKind.Array
                && modalities.EnumerateArray().Any(mode => mode.GetString() is "vision" or "image"))
                capabilities |= Capability.Vision;

            models.Add(new ModelInfo(
                item.GetProperty("id").GetString() ?? "",
                item.TryGetProperty("name", out JsonElement title) ? title.GetString() ?? "" : item.GetProperty("id").GetString() ?? "",
                RubPerMillion(pricing, "prompt_rub_per_1m"),
                RubPerMillion(pricing, "completion_rub_per_1m"),
                Number(item, "context_length"),
                Number(item, "max_completion_tokens"),
                capabilities));
        }

        return models;
    }

    // Цены FractalRouter приходят строками, иногда с запятой вместо точки, и уже за миллион токенов
    private static double RubPerMillion(JsonElement pricing, string field)
    {
        if (pricing.ValueKind != JsonValueKind.Object || !pricing.TryGetProperty(field, out JsonElement value))
            return 0;

        string text = (value.ValueKind == JsonValueKind.Number ? value.GetDouble().ToString(CultureInfo.InvariantCulture) : value.GetString() ?? "")
            .Replace(',', '.');

        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double perMillion) ? perMillion : 0;
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
}
