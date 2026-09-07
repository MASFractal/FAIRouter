using System.Text.Json;
using System.Text.Json.Serialization;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Core.Models.Common.Requests;
using FAI.Router.Enums;

namespace FAI.Router.LLM;

/// <summary>
/// Смысловая оценка текста моделью: стиль и лексические метрики,
/// то есть всё, что не считается по разметке (через OpenRouter)
/// </summary>
public class StyleClassifier
{
    /// <summary>
    /// Системный промпт оценщика
    /// </summary>
    private const string SystemPrompt =
        "Ты оцениваешь стиль и лексику присланного текста. Определи стиль, долю терминологии " +
        "и формальность тона, верни результат строго в виде JSON по заданной схеме, без пояснений.";

    private readonly string _schemaJson = BuildSchema();

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };

    /// <summary>
    /// Оценивает стиль и лексику текста (один запрос к LLM)
    /// </summary>
    /// <param name="text">Текст для оценки</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public async Task<StyleAssessment> AssessAsync(string text, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Текст для оценки не может быть пустым.", nameof(text));

        GenerateSettings settings = new(temperature: 0)
        {
            ResponseFormat = ResponseFormat.CreateJsonSchema("style_assessment", _schemaJson)
        };

        List<LLMMessage> messages =
        [
            new LLMMessage(LLMMessage.SystemRole, SystemPrompt),
            new LLMMessage(LLMMessage.UserRole, text)
        ];

        string response = await Settings.LLM.SendToLLM(messages, settings, cancellationToken).ConfigureAwait(false);
        return JsonSerializer.Deserialize<StyleAssessment>(response, JsonOptions) ?? new StyleAssessment();
    }

    // Схема ответа: имена полей совпадают со свойствами Specifications,
    // список стилей берётся из Style, чтобы не расходиться с перечислением
    private static string BuildSchema()
    {
        var schema = new
        {
            type = "object",
            properties = new
            {
                styleType = new
                {
                    type = "string",
                    @enum = Enum.GetNames<Style>(),
                    description = "Стиль текста"
                },
                termDensity = new
                {
                    type = "number",
                    description = "Доля терминологии и иностранных слов, 0-1"
                },
                formalityScore = new
                {
                    type = "number",
                    description = "Формальность тона, 0-1"
                }
            },
            required = new[] { "styleType", "termDensity", "formalityScore" },
            additionalProperties = false
        };
        return JsonSerializer.Serialize(schema);
    }
}

/// <summary>
/// Смысловые признаки текста, которые распознаёт модель
/// </summary>
/// <param name="StyleType">Стиль текста</param>
/// <param name="TermDensity">Доля терминологии, 0-1</param>
/// <param name="FormalityScore">Формальность тона, 0-1</param>
public record StyleAssessment(
    Style StyleType = Style.Other,
    double TermDensity = 0,
    double FormalityScore = 0);
