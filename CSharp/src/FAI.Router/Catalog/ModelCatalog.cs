using System.Collections.Concurrent;
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
/// <para>
/// Отрицательная цена означает, что она неизвестна: поставщик ее не сообщил, сообщил непонятно или
/// сам пометил минус единицей (так у OpenRouter помечен openrouter/auto). Ноль означает «бесплатно», и
/// путать их нельзя: прежде цена без данных становилась нулем, и модель выигрывала как бесплатная.
/// </para>
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
    /// Скорость модели, о которой ничего не известно и снимка замеров нет, токенов в секунду: обычное
    /// значение для нынешнего поколения моделей. Со снимком умолчанием служит нижняя граница серии скорости.
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

    /// <summary>Цена, которой нет: поставщик ее не сообщил</summary>
    private const double UnknownPrice = -1;

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    // Один клиент на процесс и с таймаутом: клиент на каждый вызов исчерпывал сокеты, а без
    // таймаута недоступный каталог держал создание роутера сто секунд
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(30) };

    private static readonly ConcurrentDictionary<string, (DateTimeOffset At, IReadOnlyList<ModelInfo> Models)> Cache = new();

    /// <summary>Сколько живет загруженный каталог: роутеры одного процесса не просят его заново на каждый запуск</summary>
    public static TimeSpan CacheLifetime { get; set; } = TimeSpan.FromHours(1);

    /// <summary>
    /// Загружает каталог OpenRouter, ключ не нужен. Без своего клиента ответ запоминается на
    /// <see cref="CacheLifetime"/>.
    /// </summary>
    /// <param name="client">Готовый клиент; не задан, значит общий с таймаутом</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public static async Task<IReadOnlyList<ModelInfo>> FetchAsync(HttpClient? client = null, CancellationToken cancellationToken = default)
    {
        if (client is null && Cached(OpenRouterModels) is { } cached)
            return cached;

        string json = await (client ?? Http).GetStringAsync(OpenRouterModels, cancellationToken).ConfigureAwait(false);
        IReadOnlyList<ModelInfo> models = Parse(json);

        if (client is null)
            Cache[OpenRouterModels] = (DateTimeOffset.UtcNow, models);

        return models;
    }

    /// <summary>
    /// Загружает каталог FractalRouter: цены в рублях за миллион токенов. Без своего клиента ответ
    /// запоминается на <see cref="CacheLifetime"/>.
    /// </summary>
    /// <param name="apiKey">Ключ FractalRouter, без него каталог не отдается</param>
    /// <param name="client">Готовый клиент; не задан, значит общий с таймаутом</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public static async Task<IReadOnlyList<ModelInfo>> FetchFractalRouterAsync(string apiKey, HttpClient? client = null, CancellationToken cancellationToken = default)
    {
        string cacheKey = FractalRouterModels + "\n" + apiKey;

        if (client is null && Cached(cacheKey) is { } cached)
            return cached;

        using HttpRequestMessage request = new(HttpMethod.Get, FractalRouterModels);
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        using HttpResponseMessage response = await (client ?? Http).SendAsync(request, cancellationToken).ConfigureAwait(false);
        string json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"Каталог FractalRouter не отдан: {(int)response.StatusCode} {response.ReasonPhrase}: {json}");

        IReadOnlyList<ModelInfo> models = ParseFractalRouter(json);

        if (client is null)
            Cache[cacheKey] = (DateTimeOffset.UtcNow, models);

        return models;
    }

    /// <summary>
    /// Каталог поставщика, а при сбое сети, таймауте или нечитаемом ответе популярные модели из
    /// комплекта с ценами из него (доллары OpenRouter на дату сборки). Без сети роутер прежде не
    /// создавался вовсе.
    /// </summary>
    /// <param name="fetch">Как получить каталог поставщика</param>
    /// <param name="cancellationToken">Токен отмены вызывающего: его отмена сбоем сети не считается</param>
    public static async Task<IReadOnlyList<ModelInfo>> FetchOrPopularAsync(Func<CancellationToken, Task<IReadOnlyList<ModelInfo>>> fetch, CancellationToken cancellationToken = default)
    {
        try
        {
            return await fetch(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception error) when (error is HttpRequestException or JsonException or OperationCanceledException && !cancellationToken.IsCancellationRequested)
        {
            System.Diagnostics.Trace.TraceWarning($"FAIRouter: каталог поставщика недоступен ({error.Message}); цены взяты из комплекта сборки.");
            return PopularModelInfos();
        }
    }

    /// <summary>
    /// Идентификаторы популярных моделей из комплекта: модели с внешними рейтингами в снимке замеров и
    /// недорогие рабочие лошадки
    /// </summary>
    public static IReadOnlyList<string> PopularModels() => [.. PopularModelInfos().Select(model => model.Id)];

    /// <summary>
    /// Популярные модели из комплекта со сведениями: цены в долларах за миллион токенов на дату сборки
    /// (нет цены, значит неизвестна). Окно и возможности комплект не хранит.
    /// </summary>
    public static IReadOnlyList<ModelInfo> PopularModelInfos()
    {
        using Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(PopularResourceName)
            ?? throw new InvalidOperationException($"В сборке нет ресурса {PopularResourceName}.");
        using JsonDocument document = JsonDocument.Parse(stream);

        return [.. Items(document.RootElement, "models")
            .Where(item => Text(item, "id") is { Length: > 0 })
            .Select(item => new ModelInfo(
                Text(item, "id")!,
                Text(item, "title") ?? "",
                Decimal(item, "usd_per_million_input") ?? UnknownPrice,
                Decimal(item, "usd_per_million_output") ?? UnknownPrice,
                0,
                0,
                Capability.Code | Capability.Formulas | Capability.Tools))];
    }

    /// <summary>
    /// Список идентификаторов по тому, что передали: один элемент «all» дает весь каталог поставщика
    /// с известной ценой, «popular» дает популярные из комплекта, которые есть в каталоге, иной список
    /// берется как есть, без повторов. Strict говорит, обязан ли каждый идентификатор найтись: у
    /// именованного набора отсутствующие модели молча пропускаются, у списка их отсутствие это ошибка.
    /// </summary>
    /// <param name="modelIds">Список идентификаторов либо имя набора одним элементом; пусто, значит popular</param>
    /// <param name="known">Каталог поставщика по идентификаторам</param>
    public static (IReadOnlyList<string> Ids, bool Strict) Select(IEnumerable<string>? modelIds, IReadOnlyDictionary<string, ModelInfo> known)
    {
        string[] ids = modelIds is null ? [Popular] : [.. modelIds.Distinct()];

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
    /// Делает кандидата на исполнение из сведений каталога; начальный вектор строится в пространстве
    /// общего среднего <see cref="Settings.TaskMean"/>
    /// </summary>
    /// <param name="model">Сведения о модели</param>
    /// <param name="tokensPerSecond">Скорость, которой в каталоге нет; не задана, тогда из рейтингов</param>
    /// <param name="benchmarks">Снимок рейтингов для начального вектора; пусто, значит вектор случайный</param>
    public static BaseRoutedElement CreateElement(ModelInfo model, double? tokensPerSecond = null, BenchmarkSnapshot? benchmarks = null) =>
        CreateElement(model, tokensPerSecond, benchmarks, Settings.TaskMean);

    /// <summary>
    /// Делает кандидата на исполнение из сведений каталога
    /// </summary>
    /// <remarks>
    /// Со снимком рейтингов кандидат стартует не со случайного вектора, а с прогноза по сериям внешних
    /// замеров (<see cref="BenchmarkPrior"/>), и это стоит условного опыта
    /// (<see cref="BaseRoutedElement.PriorExperience"/>). Модель, которой в рейтингах нет, получает
    /// уровень поля (<see cref="BenchmarkPrior.FieldQuality"/>) во всех сериях: прежде она получала
    /// случайный вектор и побеждала или проигрывала наугад. Оттуда же берутся скорость, если ее не
    /// назвали (нет в серии, значит нижняя граница серии), и поправка цены на рассуждения.
    /// </remarks>
    /// <param name="model">Сведения о модели</param>
    /// <param name="tokensPerSecond">Скорость, которой в каталоге нет; не задана, тогда из рейтингов</param>
    /// <param name="benchmarks">Снимок рейтингов для начального вектора; пусто, значит вектор случайный</param>
    /// <param name="taskMean">Среднее задач, в пространстве которого строится вектор; пусто, значит без вычитания</param>
    public static BaseRoutedElement CreateElement(ModelInfo model, double? tokensPerSecond, BenchmarkSnapshot? benchmarks, Vector? taskMean)
    {
        BaseRoutedElement element = new()
        {
            Name = model.Id,
            DPMTInp = PriceOrUnknown(model.DollarsPerMillionInput),
            DPMTOutp = PriceOrUnknown(model.DollarsPerMillionOutput),
            TPS = tokensPerSecond
                ?? (benchmarks is null ? null : BenchmarkPrior.TokensPerSecond(benchmarks, model.Id) ?? BenchmarkPrior.DefaultTokensPerSecond(benchmarks))
                ?? DefaultTokensPerSecond,
            Capabilities = model.Capabilities,
            TaskMean = taskMean,

            // ContextLimit это предел ОТВЕТА в символах, а поставщик считает в токенах; окно в токенах
            ContextLimit = (int)Math.Min(int.MaxValue, Math.Max(model.MaxAnswerTokens, 0) * InputFeaturesService.EST_SYMBOL_PER_TOKEN),
            ContextWindow = Math.Max(model.ContextTokens, 0)
        };

        if (benchmarks is null)
            return element;

        if (BenchmarkPrior.GetVector(benchmarks, model.Id, taskMean) is { } prior)
        {
            element.IdealMatchVector = prior;
            element.PriorExperience = Settings.PriorExperience;
        }
        else if (BenchmarkPrior.FieldQuality(benchmarks) is { } field)
            element.IdealMatchVector = BenchmarkPrior.Uniform(field, taskMean);

        element.CostRatio = BenchmarkPrior.CostRatio(benchmarks, model.Id) ?? element.CostRatio;

        return element;
    }

    /// <summary>
    /// Кандидаты по идентификаторам: цена из <paramref name="prices"/> поверх каталога, остальное из
    /// каталога. Модель только из prices считается способной на все, кроме того, что опровергает
    /// каталог; модели без цены ни там, ни там быть не может.
    /// </summary>
    /// <param name="ids">Идентификаторы без повторов</param>
    /// <param name="known">Каталог поставщика</param>
    /// <param name="prices">Цены за миллион токенов для моделей, которых нет в каталоге или чью цену надо заменить</param>
    /// <param name="tokensPerSecond">Скорость по идентификаторам</param>
    /// <param name="benchmarks">Снимок замеров</param>
    /// <param name="taskMean">Среднее задач для начальных векторов</param>
    public static List<BaseRoutedElement> CreateCandidates(
        IEnumerable<string> ids,
        IReadOnlyDictionary<string, ModelInfo> known,
        IReadOnlyDictionary<string, Price> prices,
        IReadOnlyDictionary<string, double>? tokensPerSecond,
        BenchmarkSnapshot? benchmarks,
        Vector? taskMean)
    {
        List<BaseRoutedElement> candidates = [];

        foreach (string id in ids.Distinct())
        {
            ModelInfo? found = known.GetValueOrDefault(id);
            ModelInfo info = prices.TryGetValue(id, out Price? price)
                ? (found ?? new ModelInfo(id, id, 0, 0, 0, 0, Capability.All)) with { DollarsPerMillionInput = price.Input, DollarsPerMillionOutput = price.Output }
                : found ?? throw new ArgumentException(
                    $"У модели {id} нет цены в prices, и в каталоге поставщика ее нет. "
                    + $"Задайте цену за миллион токенов: prices[\"{id}\"] = new Price(вход, выход).", nameof(ids));

            if (info.DollarsPerMillionInput < 0 || info.DollarsPerMillionOutput < 0)
                System.Diagnostics.Trace.TraceWarning($"FAIRouter: цена модели {id} неизвестна, в выборе она получает худшую цену группы.");

            double? speed = tokensPerSecond is not null && tokensPerSecond.TryGetValue(id, out double tps) ? tps : null;
            candidates.Add(CreateElement(info, speed, benchmarks, taskMean));
        }

        return candidates;
    }

    /// <summary>
    /// Разбирает ответ каталога OpenRouter. Битая запись пропускается, а не роняет весь каталог.
    /// </summary>
    /// <param name="json">Тело ответа</param>
    public static IReadOnlyList<ModelInfo> Parse(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        List<ModelInfo> models = [];

        foreach (JsonElement item in Items(document.RootElement, "data"))
        {
            if (Text(item, "id") is not { Length: > 0 } id)
                continue;

            JsonElement pricing = Property(item, "pricing");

            models.Add(new ModelInfo(
                id,
                Text(item, "name") ?? "",
                PerMillion(pricing, "prompt"),
                PerMillion(pricing, "completion"),
                Number(item, "context_length"),
                Number(Property(item, "top_provider"), "max_completion_tokens"),
                CapabilitiesOf(item, pricing)));
        }

        return models;
    }

    /// <summary>
    /// Разбирает ответ каталога FractalRouter: цены уже за миллион токенов, в рублях. Предел ответа
    /// каталог не сообщает, поэтому ограничения по объему у кандидата нет. Битая запись пропускается.
    /// </summary>
    /// <param name="json">Тело ответа</param>
    public static IReadOnlyList<ModelInfo> ParseFractalRouter(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        List<ModelInfo> models = [];

        foreach (JsonElement item in root.ValueKind == JsonValueKind.Array ? Items(root) : Items(root, "data"))
        {
            if (Text(item, "id") is not { Length: > 0 } id)
                continue;

            JsonElement pricing = Property(item, "pricing");

            // Умение вызывать инструменты каталог не сообщает; считаем, что умеют все, иначе запрос
            // с required: Capability.Tools не нашел бы ни одного кандидата
            Capability capabilities = Capability.Code | Capability.Formulas | Capability.Tools;

            if (Strings(item, "modalities").Any(mode => mode is "vision" or "image"))
                capabilities |= Capability.Vision;

            models.Add(new ModelInfo(
                id,
                Text(item, "name") ?? id,
                Decimal(pricing, "prompt_rub_per_1m") ?? UnknownPrice,
                Decimal(pricing, "completion_rub_per_1m") ?? UnknownPrice,
                Number(item, "context_length"),
                Number(item, "max_completion_tokens"),
                capabilities));
        }

        return models;
    }

    private static IReadOnlyList<ModelInfo>? Cached(string key) =>
        Cache.TryGetValue(key, out var entry) && DateTimeOffset.UtcNow - entry.At < CacheLifetime ? entry.Models : null;

    // Отрицательная цена любого вида приводится к одной метке «неизвестна»
    private static double PriceOrUnknown(double price) => double.IsFinite(price) && price >= 0 ? price : UnknownPrice;

    // Цены OpenRouter приходят строками и за один токен; нет цены или она отрицательна, значит неизвестна
    private static double PerMillion(JsonElement pricing, string field) =>
        Decimal(pricing, field) is >= 0 and var perToken ? perToken * 1e6 : UnknownPrice;

    private static Capability CapabilitiesOf(JsonElement item, JsonElement pricing)
    {
        // Писать код и формулы умеет любая текстовая модель, это не отличительная черта
        Capability capabilities = Capability.Code | Capability.Formulas;

        if (Strings(Property(item, "architecture"), "input_modalities").Contains("image"))
            capabilities |= Capability.Vision;

        if (Strings(item, "supported_parameters").Contains("tools"))
            capabilities |= Capability.Tools;

        // Свой поиск в сети у модели есть, если поставщик назначил ему цену
        if (Decimal(pricing, "web_search") is > 0)
            capabilities |= Capability.WebSearch;

        return capabilities;
    }

    private static JsonElement Property(JsonElement item, string field) =>
        item.ValueKind == JsonValueKind.Object && item.TryGetProperty(field, out JsonElement value) ? value : default;

    private static IEnumerable<JsonElement> Items(JsonElement root, string? field = null)
    {
        JsonElement items = field is null ? root : Property(root, field);

        return items.ValueKind == JsonValueKind.Array ? items.EnumerateArray() : [];
    }

    private static IEnumerable<string> Strings(JsonElement item, string field) =>
        Items(item, field).Where(value => value.ValueKind == JsonValueKind.String).Select(value => value.GetString()!);

    private static string? Text(JsonElement item, string field)
    {
        JsonElement value = Property(item, field);

        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Number => value.GetRawText(),
            _ => null
        };
    }

    // Число из числа или строки; строки бывают с запятой вместо точки
    private static double? Decimal(JsonElement item, string field) =>
        Text(item, field)?.Replace(',', '.') is { } text
        && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number)
        && double.IsFinite(number)
            ? number
            : null;

    private static int Number(JsonElement item, string field) =>
        Decimal(item, field) is { } number ? (int)Math.Clamp(number, 0, int.MaxValue) : 0;
}
