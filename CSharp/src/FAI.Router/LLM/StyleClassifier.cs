using System.Text.Json;
using System.Text.Json.Serialization;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Services.LLM;
using FAI.Router.Enums;

namespace FAI.Router.LLM;

/// <summary>
/// Смысловая оценка текста моделью: стиль и лексические метрики,
/// то есть все, что не считается по разметке (через OpenRouter)
/// </summary>
/// <remarks>
/// Отдельное от судьи содержания обращение оставлено намеренно: стиль и предмет ответа модель
/// оценивает по одному ответу, не видя задания. Судья видит задание, и в общем вызове предмет
/// ответа списывался бы с предмета задания, а сверка «ответ о том же, о чем задание» потеряла бы
/// смысл. Ответ обрезается тем же пределом, что у судьи: прежде уходил целиком (бывало 62 925 знаков).
/// </remarks>
public class StyleClassifier
{
    /// <summary>
    /// Системный промпт оценщика
    /// </summary>
    private const string SystemPrompt =
        "Ты оцениваешь стиль, лексику и предмет присланного текста. Определи стиль, долю терминологии, " +
        "формальность тона, предметную область, область науки и тип результата (что это за текст), верни результат " +
        "строго в виде JSON по заданной схеме, без пояснений.";

    private static readonly string SchemaJson = BuildSchema();

    // Число вместо имени значения не принимается: styleType 42 выходил за границы перечисления
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter(allowIntegerValues: false) }
    };

    private readonly LLMBase? _llm;

    /// <summary>Бюджет времени на оценку, все попытки вместе</summary>
    public TimeSpan Budget { get; init; } = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Смысловая оценка текста моделью
    /// </summary>
    /// <param name="llm">
    /// Клиент модели. Не задан, тогда берется общий Settings.LLM. Свой клиент нужен, когда в одном
    /// процессе судят несколько моделей: общий клиент один, и подставлять его по очереди значит
    /// запретить параллельное сравнение.
    /// </param>
    public StyleClassifier(LLMBase? llm = null) => _llm = llm;

    /// <summary>
    /// Оценивает стиль и лексику текста (один запрос к LLM)
    /// </summary>
    /// <param name="text">Текст для оценки</param>
    /// <param name="cancellationToken">Токен отмены</param>
    /// <exception cref="InvalidDataException">Модель дважды ответила не по схеме</exception>
    public async Task<StyleAssessment> AssessAsync(string text, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Текст для оценки не может быть пустым.", nameof(text));

        string tag = PromptData.NewTag();
        List<LLMMessage> messages =
        [
            new LLMMessage(LLMMessage.SystemRole, SystemPrompt + " " + PromptData.Rule(tag)),
            new LLMMessage(LLMMessage.UserRole, PromptData.Wrap(tag, "текст", PromptData.Clip(text, ContentJudge.AnswerChars)))
        ];

        return await JsonCall.AskAsync(_llm ?? Settings.LLM, messages, JsonCall.Settings("style_assessment", SchemaJson),
            Read, Budget, cancellationToken).ConfigureAwait(false);
    }

    // Значение вне перечисления негодно: оно вышло бы за границы кода «один из многих»
    private static StyleAssessment? Read(string json) =>
        JsonSerializer.Deserialize<StyleAssessment>(json, JsonOptions) is { } assessment
        && Enum.IsDefined(assessment.StyleType) && Enum.IsDefined(assessment.Domain)
        && Enum.IsDefined(assessment.ScienceField) && Enum.IsDefined(assessment.TaskKind)
        && double.IsFinite(assessment.TermDensity) && double.IsFinite(assessment.FormalityScore)
            ? assessment
            : null;

    // Схема ответа: имена полей совпадают со свойствами Specifications,
    // список стилей берется из Style, чтобы не расходиться с перечислением
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
                    minimum = 0,
                    maximum = 1,
                    description = SpecFieldDescriptions.TermDensity
                },
                formalityScore = new
                {
                    type = "number",
                    minimum = 0,
                    maximum = 1,
                    description = "Формальность тона, 0-1"
                },
                domain = new { type = "string", @enum = Enum.GetNames<Domain>(), description = SpecFieldDescriptions.Domain },
                scienceField = new { type = "string", @enum = Enum.GetNames<ScienceField>(), description = SpecFieldDescriptions.ScienceField },
                taskKind = new { type = "string", @enum = Enum.GetNames<TaskKind>(), description = SpecFieldDescriptions.TaskKind }
            },
            required = new[] { "styleType", "termDensity", "formalityScore", "domain", "scienceField", "taskKind" },
            additionalProperties = false
        };
        return JsonSerializer.Serialize(schema);
    }
}

/// <summary>
/// Смысловые признаки текста, которые распознает модель
/// </summary>
/// <param name="StyleType">Стиль текста</param>
/// <param name="TermDensity">Доля терминологии, 0-1</param>
/// <param name="FormalityScore">Формальность тона, 0-1</param>
/// <param name="Domain">Предметная область</param>
/// <param name="ScienceField">Область науки</param>
/// <param name="TaskKind">Тип результата</param>
public record StyleAssessment(
    Style StyleType = Style.Other,
    double TermDensity = 0,
    double FormalityScore = 0,
    Domain Domain = Domain.General,
    ScienceField ScienceField = ScienceField.None,
    TaskKind TaskKind = TaskKind.None);
