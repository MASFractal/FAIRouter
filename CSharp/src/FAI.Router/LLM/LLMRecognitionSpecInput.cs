using System.Text.Json;
using System.Text.Json.Serialization;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Core.Models.Common.Requests;
using AI.LLM.Services.LLM;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;

namespace FAI.Router.LLM;

/// <summary>
/// Получение спецификации на входе (через LLM)
/// </summary>
/// <remarks>
/// Запрос пользователя идет модели внутри метки со случайным именем (<see cref="PromptData"/>):
/// ограду из черточек закрывала та же строка черточек в запросе. Ответ проверяется: значение вне
/// перечисления (в том числе число вместо имени), пустые пункты и неизвестные имена полей
/// отбрасываются, а негодный ответ повторяется один раз и затем это сбой распознавания.
/// </remarks>
public class LLMRecognitionSpecInput
{
    /// <summary>
    /// Роль модели: извлечь из запроса требования к будущему ответу
    /// </summary>
    private const string SystemPrompt = """
        Извлеки техническое задание из запроса пользователя к языковой модели.
        Опиши, каким должен быть ОТВЕТ на этот запрос, и верни параметры ответа в JSON по схеме.
        Явно заданные требования (объем, стиль, число разделов, таблицы, язык) бери как есть и перечисли имена этих полей в explicitFields.
        Неуказанное оценивай разумным ожиданием для такой задачи, а не нулем, но в explicitFields не вноси: угаданное требованием не считается.
        Язык ответа задан явно, если он назван или если запрос написан на нем и другой язык не назван.
        Отдельно выпиши смысловые пункты, которые ответ обязан раскрыть, и явные ограничения запроса.
        Область определяй по функции, которой служит результат: работа с покупателем и продажи (коммерческое предложение, письмо клиенту о ценах, скрипт звонка) это Sales, продвижение и реклама это Marketing, персонал это Hr, склад, закупки и возвраты это Operations, налоги и отчетность это Finance. Business ставь только для стратегии и управления компанией в целом.
        """;

    private static readonly string SchemaJson = BuildSchema();

    // Число вместо имени значения не принимается: styleType 42 выходил за границы перечисления
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter(allowIntegerValues: false) }
    };

    private readonly GenerateSettings _settings = JsonCall.Settings("input_specifications", SchemaJson);
    private readonly LLMBase? _llm;

    /// <summary>
    /// Бюджет времени на распознавание, все попытки вместе: ход ждет его до выбора исполнителя
    /// </summary>
    public TimeSpan Budget { get; init; } = TimeSpan.FromMinutes(1);

    /// <summary>
    /// Получение спецификации на входе (через LLM)
    /// </summary>
    /// <param name="llm">Клиент модели; не задан, тогда берется общий Settings.LLM</param>
    public LLMRecognitionSpecInput(LLMBase? llm = null) => _llm = llm;

    /// <summary>
    /// Извлекает ожидаемую спецификацию ответа из текста запроса (один запрос к LLM, при негодном
    /// ответе еще один)
    /// </summary>
    /// <param name="prompt">Текст запроса пользователя</param>
    /// <param name="cancellationToken">Токен отмены</param>
    /// <exception cref="InvalidDataException">Модель дважды ответила не по схеме</exception>
    /// <exception cref="TimeoutException">Модель не уложилась в <see cref="Budget"/></exception>
    public async Task<Specifications> GetSpecificationsAsync(string prompt, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("Запрос не может быть пустым.", nameof(prompt));

        // Сообщения собираются явно: перегрузка SendToLLM(string) подставляет системным сообщением
        // промпт КЛИЕНТА, а он у общего Settings.LLM не задан, поэтому поставщик отвергает content = null
        string tag = PromptData.NewTag();
        List<LLMMessage> messages =
        [
            new LLMMessage(LLMMessage.SystemRole, SystemPrompt + "\n" + PromptData.Rule(tag)),
            new LLMMessage(LLMMessage.UserRole, PromptData.Wrap(tag, "запрос пользователя", prompt))
        ];

        return await JsonCall.AskAsync(_llm ?? Settings.LLM, messages, _settings, Read, Budget, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Разбор ответа модели: значения перечислений только из перечислений, пункты без пустых и не
    /// больше <see cref="Specifications.MaxItems"/>, язык кодом без региона, явные поля только из
    /// <see cref="Specifications.StatableFields"/>. Пусто, если ответ негоден.
    /// </summary>
    /// <param name="json">Ответ модели по схеме</param>
    internal static Specifications? Read(string json)
    {
        Specifications? spec = JsonSerializer.Deserialize<Specifications>(json, JsonOptions);

        if (spec is null || !Enum.IsDefined(spec.StyleType) || !Enum.IsDefined(spec.Domain) || !Enum.IsDefined(spec.ProgrammingLanguage)
            || !Enum.IsDefined(spec.ScienceField) || !Enum.IsDefined(spec.TaskKind) || !double.IsFinite(spec.AvgSentenceLength))
            return null;

        spec.RequiredPoints = Items(spec.RequiredPoints);
        spec.Constraints = Items(spec.Constraints);
        spec.Language = Specifications.NormalizeLanguage(spec.Language);

        if (spec.ExplicitFields is not null)
            spec.ExplicitFields = [.. Specifications.StatableFields.Where(field => spec.ExplicitFields.Contains(field, StringComparer.OrdinalIgnoreCase))];

        return spec;
    }

    // Пункты без пустых и повторов, не больше предела: null в списке и сам список null модель
    // присылала, а дальше по ним шел суд
    private static List<string> Items(List<string>? items) =>
        [.. (items ?? []).Where(item => !string.IsNullOrWhiteSpace(item)).Select(item => item.Trim()).Distinct().Take(Specifications.MaxItems)];

    // Схема ответа: имена полей совпадают со свойствами Specifications,
    // список стилей берется из Style, чтобы не расходиться с перечислением
    private static string BuildSchema()
    {
        string styles = Names(Enum.GetNames<Style>());
        string domains = Names(Enum.GetNames<Domain>());
        string languages = Names(Enum.GetNames<ProgrammingLanguage>());
        string sciences = Names(Enum.GetNames<ScienceField>());
        string kinds = Names(Enum.GetNames<TaskKind>());
        string speech = Names([.. Specifications.LanguageCodes, "other"]);
        string statable = Names(Specifications.StatableFields);

        return $$"""
            {
              "type": "object",
              "properties": {
                "styleType": { "type": "string", "enum": [{{styles}}], "description": "Стиль текста ответа" },
                "symbolLength": { "type": "integer", "description": "Объем ответа в символах" },
                "wordLength": { "type": "integer", "description": "Объем ответа в словах" },
                "paragraphCount": { "type": "integer", "description": "Число абзацев" },
                "sectionCount": { "type": "integer", "description": "Число разделов" },
                "listItemCount": { "type": "integer", "description": "Число пунктов списков" },
                "tableCount": { "type": "integer", "description": "Число таблиц" },
                "codeBlockCount": { "type": "integer", "description": "Число блоков кода" },
                "formulaCount": { "type": "integer", "description": "Число формул" },
                "headingDepth": { "type": "integer", "description": "Глубина вложенности заголовков" },
                "avgSentenceLength": { "type": "number", "description": "Средняя длина предложения в словах" },
                "readabilityScore": { "type": "number", "minimum": 0, "maximum": 100, "description": "{{SpecFieldDescriptions.Readability}}" },
                "termDensity": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.TermDensity}}" },
                "formalityScore": { "type": "number", "minimum": 0, "maximum": 1, "description": "Формальность тона, 0-1" },
                "language": { "type": "string", "enum": [{{speech}}], "description": "{{SpecFieldDescriptions.Language}}" },
                "hasReferences": { "type": "boolean", "description": "Нужны ли ссылки на источники" },
                "domain": { "type": "string", "enum": [{{domains}}], "description": "{{SpecFieldDescriptions.Domain}}" },
                "programmingLanguage": { "type": "string", "enum": [{{languages}}], "description": "{{SpecFieldDescriptions.ProgrammingLanguage}}" },
                "scienceField": { "type": "string", "enum": [{{sciences}}], "description": "{{SpecFieldDescriptions.ScienceField}}" },
                "taskKind": { "type": "string", "enum": [{{kinds}}], "description": "{{SpecFieldDescriptions.TaskKind}}" },
                "expertLevel": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.ExpertLevel}}" },
                "difficulty": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.Difficulty}}" },
                "factualityDemand": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.FactualityDemand}}" },
                "requiredPoints": { "type": "array", "items": { "type": "string" }, "description": "{{SpecFieldDescriptions.RequiredPoints}}" },
                "constraints": { "type": "array", "items": { "type": "string" }, "description": "{{SpecFieldDescriptions.Constraints}}" },
                "explicitFields": { "type": "array", "items": { "type": "string", "enum": [{{statable}}] }, "description": "{{SpecFieldDescriptions.ExplicitFields}}" }
              },
              "required": [
                "styleType", "symbolLength", "wordLength", "paragraphCount", "sectionCount",
                "listItemCount", "tableCount", "codeBlockCount", "formulaCount", "headingDepth",
                "avgSentenceLength", "readabilityScore", "termDensity", "formalityScore",
                "language", "hasReferences", "domain", "programmingLanguage", "scienceField", "taskKind",
                "expertLevel", "difficulty", "factualityDemand", "requiredPoints", "constraints", "explicitFields"
              ],
              "additionalProperties": false
            }
            """;
    }

    // Список значений для схемы, в кавычках через запятую
    private static string Names(IEnumerable<string> names) => string.Join(", ", names.Select(name => $"\"{name}\""));
}
