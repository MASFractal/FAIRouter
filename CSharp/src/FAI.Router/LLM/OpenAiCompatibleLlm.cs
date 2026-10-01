using AI.LLM.Clients.Base;
using AI.LLM.Clients.OpenRouter;
using AI.LLM.Services.LLM;

namespace FAI.Router.LLM;

/// <summary>
/// Адреса поставщиков по протоколу OpenAI chat completions
/// </summary>
/// <remarks>
/// FractalRouter (fractalrouter.ru) это наш шлюз к моделям с оплатой в рублях, не путать с FractalGPT.
/// Любой другой поставщик с тем же протоколом подключается через <see cref="OpenAiCompatibleLlm"/>
/// с его адресом.
/// </remarks>
public static class Providers
{
    /// <summary>
    /// FractalRouter: ключи вида rtr_live_..., каталог моделей с ценами в рублях
    /// </summary>
    public const string FractalRouter = "https://api.fractalrouter.ru/v1";

    /// <summary>
    /// OpenRouter: каталог с ценами в долларах
    /// </summary>
    public const string OpenRouter = "https://openrouter.ai/api/v1";

    /// <summary>
    /// Поставщик по виду ключа: ключи OpenRouter начинаются с sk-or-, ключи FractalRouter с rtr_live_
    /// или frr_test_. Незнакомый ключ считается ключом FractalRouter.
    /// </summary>
    /// <param name="apiKey">Ключ поставщика</param>
    public static string ForKey(string apiKey) =>
        apiKey.StartsWith("sk-or-", StringComparison.Ordinal) ? OpenRouter : FractalRouter;

    /// <summary>
    /// Поставщик и ключ из окружения: переменная FRACTALROUTER_API_KEY дает FractalRouter,
    /// OPENROUTER_API_KEY дает OpenRouter, иначе ключ читается из файла key.txt в указанных каталогах
    /// и поставщик опознается по виду ключа. Пустой ключ означает, что ничего не найдено.
    /// </summary>
    /// <param name="keyDirectories">Где искать key.txt, по порядку</param>
    public static (string BaseUrl, string ApiKey) FromEnvironment(params string[] keyDirectories)
    {
        string? fractal = Environment.GetEnvironmentVariable("FRACTALROUTER_API_KEY");
        if (!string.IsNullOrWhiteSpace(fractal))
            return (FractalRouter, fractal.Trim());

        string? open = Environment.GetEnvironmentVariable("OPENROUTER_API_KEY");
        if (!string.IsNullOrWhiteSpace(open))
            return (OpenRouter, open.Trim());

        foreach (string directory in keyDirectories)
        {
            string path = Path.Combine(directory, "key.txt");
            if (!File.Exists(path))
                continue;

            string key = File.ReadAllText(path).Trim();
            if (key.Length > 0)
                return (ForKey(key), key);
        }

        return (FractalRouter, "");
    }
}

/// <summary>
/// Клиент модели у любого поставщика по протоколу OpenAI chat completions: FractalRouter, OpenRouter,
/// vLLM и прочие. Отличаются они только адресом, поэтому класс один, а адрес задается строкой.
/// </summary>
/// <remarks>
/// Построен на клиенте OpenRouter из AI.LLM: тот не проверяет размер контекста перед отправкой, и
/// это нужно для всех шлюзов, у которых окно модели заранее неизвестно. То же, что OpenRouterClient
/// с base_url в версии на Python.
/// </remarks>
public class OpenAiCompatibleLlm : LLMBase
{
    /// <summary>
    /// Клиент модели у поставщика по адресу
    /// </summary>
    /// <param name="baseUrl">Адрес поставщика вида https://host/v1, см. <see cref="Providers"/></param>
    /// <param name="apiKey">Ключ поставщика, уходит заголовком Bearer</param>
    /// <param name="model">Идентификатор модели у поставщика</param>
    /// <param name="systemPrompt">Системная подсказка для запросов без контекста</param>
    public OpenAiCompatibleLlm(string baseUrl, string apiKey, string model, string systemPrompt = "")
        : base(Init(baseUrl, apiKey, model, systemPrompt), new LLMOptions { ApiKey = apiKey, ModelName = model, SystemPrompt = systemPrompt })
    {
        BaseUrl = baseUrl.TrimEnd('/');
        Model = model;
    }

    /// <summary>
    /// Адрес поставщика без завершающей косой черты
    /// </summary>
    public string BaseUrl { get; }

    /// <summary>
    /// Идентификатор модели
    /// </summary>
    public string Model { get; }

    private static ChatLLMApi Init(string baseUrl, string apiKey, string model, string systemPrompt)
    {
        if (string.IsNullOrWhiteSpace(baseUrl))
            throw new ArgumentException("Адрес поставщика не задан.", nameof(baseUrl));

        return new OpenRouterModelApi(apiKey, model, systemPrompt)
        {
            ApiUrl = $"{baseUrl.TrimEnd('/')}/chat/completions"
        };
    }
}
