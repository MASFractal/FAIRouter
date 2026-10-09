using System.Text.Json;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Core.Models.Common.Requests;
using AI.LLM.Core.Models.Common.Responses;
using AI.LLM.Services.LLM;

namespace FAI.Router.LLM;

/// <summary>
/// Обращение к модели за ответом по схеме: свой бюджет времени, явный потолок ответа, один повтор
/// при обрезке или негодном ответе и честная отмена. Общее у распознавания задания, оценки стиля и
/// судьи содержания.
/// </summary>
/// <remarks>
/// Движок (ChatLLMApi) по умолчанию ставит потолок 3012 токенов, обрезку по длине считает нормой,
/// отмену вызывающего на второй попытке отдает обычным исключением, а таймаут у него 18 минут на
/// попытку. Здесь все это решается на своей стороне: потолок задан явно, обрезка и ответ не по схеме
/// это сбой с одним повтором, отмена вызывающего всегда <see cref="OperationCanceledException"/>,
/// исчерпанный бюджет всегда <see cref="TimeoutException"/>, отказ поставщика (4xx) не повторяется.
/// </remarks>
internal static class JsonCall
{
    /// <summary>Потолок ответа модели в токенах: разбор длинного ответа по схеме в 3012 не влезал</summary>
    public const int MaxTokens = 8000;

    private const int Attempts = 2;

    /// <summary>Настройки обращения по схеме: температура ноль и явный потолок ответа</summary>
    /// <param name="name">Имя схемы</param>
    /// <param name="schemaJson">Схема ответа</param>
    public static GenerateSettings Settings(string name, string schemaJson) =>
        new(temperature: 0, maxTokens: MaxTokens) { ResponseFormat = ResponseFormat.CreateJsonSchema(name, schemaJson) };

    /// <summary>
    /// Ответ модели, разобранный по схеме. Обрезка по потолку и ответ, который разбор отверг,
    /// повторяются один раз, затем это <see cref="InvalidDataException"/>.
    /// </summary>
    /// <param name="llm">Клиент модели</param>
    /// <param name="messages">Сообщения</param>
    /// <param name="settings">Настройки обращения</param>
    /// <param name="parse">Разбор JSON: пусто, если ответ не годится</param>
    /// <param name="budget">Бюджет времени на все попытки</param>
    /// <param name="cancellationToken">Токен отмены вызывающего</param>
    public static async Task<T> AskAsync<T>(
        LLMBase llm, IReadOnlyList<LLMMessage> messages, GenerateSettings settings, Func<string, T?> parse,
        TimeSpan budget, CancellationToken cancellationToken) where T : class
    {
        using CancellationTokenSource deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(budget);
        string failure = "";

        for (int attempt = 0; attempt < Attempts; attempt++)
        {
            Choice? choice = (await SendAsync(llm, messages, settings, budget, deadline, cancellationToken).ConfigureAwait(false))
                .Choices?.FirstOrDefault();

            if (IsCut(choice))
            {
                failure = "ответ обрезан по потолку токенов";
                continue;
            }

            if (Parse(choice?.Message?.Content?.ToString(), parse) is { } result)
                return result;

            failure = "ответ не по схеме";
        }

        throw new InvalidDataException($"Модель ответила негодно на обеих попытках: {failure}.");
    }

    /// <summary>
    /// JSON из ответа модели: без ограды ```json и текста вокруг. Пусто, если фигурных скобок нет.
    /// </summary>
    /// <param name="text">Ответ модели</param>
    public static string? ExtractJson(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        int start = text.IndexOf('{');
        int end = text.LastIndexOf('}');

        return start >= 0 && end > start ? text[start..(end + 1)] : null;
    }

    /// <summary>
    /// Разбор ответа: вырезанный JSON отдается разбору, его ошибка формата означает негодный ответ
    /// </summary>
    /// <param name="text">Ответ модели</param>
    /// <param name="parse">Разбор JSON</param>
    public static T? Parse<T>(string? text, Func<string, T?> parse) where T : class
    {
        if (ExtractJson(text) is not { } json)
            return null;

        try
        {
            return parse(json);
        }
        catch (Exception error) when (error is JsonException or FormatException or InvalidDataException)
        {
            return null;
        }
    }

    // Отмена вызывающего пробрасывается отменой, исчерпанный бюджет таймаутом, отказ поставщика
    // своим исключением без обертки движка: в обертке движка текст запроса пользователя
    private static async Task<ChatCompletionsResponse> SendAsync(
        LLMBase llm, IReadOnlyList<LLMMessage> messages, GenerateSettings settings, TimeSpan budget,
        CancellationTokenSource deadline, CancellationToken cancellationToken)
    {
        try
        {
            return await llm.SendToLLMFull(messages, settings, deadline.Token).ConfigureAwait(false);
        }
        catch (Exception error) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException("Обращение к модели отменено вызывающим.", error, cancellationToken);
        }
        catch (Exception) when (deadline.IsCancellationRequested)
        {
            throw new TimeoutException($"Модель не ответила за {budget.TotalSeconds:0} с.");
        }
        catch (Exception error) when (ProviderRejectedException.Find(error) is { } rejected)
        {
            throw new ProviderRejectedException(rejected.Code, rejected.ProviderMessage);
        }
    }

    // Обрезка по потолку: у OpenAI-совместимых length, у Gemini MAX_TOKENS
    private static bool IsCut(Choice? choice) =>
        string.Equals(choice?.FinishReason, "length", StringComparison.OrdinalIgnoreCase)
        || string.Equals(choice?.NativeFinishReason, "length", StringComparison.OrdinalIgnoreCase)
        || string.Equals(choice?.NativeFinishReason, "MAX_TOKENS", StringComparison.OrdinalIgnoreCase);
}
