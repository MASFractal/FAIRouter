using AI.LLM.Core.Models.Common.Requests;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Services.LLM;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.Services;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace FAI.Router.LLM;

/// <summary>
/// Получение спецификации на входе (через LLM)
/// </summary>
public class LLMRecognitionSpecInput
{
    /// <summary>
    /// Роль модели: извлечь из запроса требования к будущему ответу
    /// </summary>
    private const string PromptTmpl = """
        Извлеки техническое задание из запроса пользователя к языковой модели.
        Опиши, каким должен быть ОТВЕТ на этот запрос, и верни параметры ответа в JSON по схеме.
        Явно заданные требования (объем, стиль, число разделов, таблицы, язык) бери как есть.
        Неуказанное оценивай разумным ожиданием для такой задачи, а не нулем.
        Отдельно выпиши смысловые пункты, которые ответ обязан раскрыть, и явные ограничения запроса.

        Запрос пользователя:
        ----
        {text}
        ----
        """;

    private readonly GenerateSettings _settings;
    private readonly LLMBase? _llm;

    private static readonly string SchemaJson = BuildSchema();

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };

    /// <summary>
    /// Получение спецификации на входе (через LLM)
    /// </summary>
    /// <param name="llm">Клиент модели; не задан, тогда берется общий Settings.LLM</param>
    public LLMRecognitionSpecInput(LLMBase? llm = null)
    {
        _llm = llm;
        _settings = new(temperature: 0)
        {
            ResponseFormat = ResponseFormat.CreateJsonSchema("input_specifications", SchemaJson)
        };
    }

    /// <summary>
    /// Извлекает ожидаемую спецификацию ответа из текста запроса (один запрос к LLM)
    /// </summary>
    /// <param name="prompt">Текст запроса пользователя</param>
    public async Task<Specifications> GetSpecificationsAsync(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("Запрос не может быть пустым.", nameof(prompt));

        string promptForModel = PromptTmpl.Replace("{text}", prompt);

        // Сообщение собирается явно: перегрузка SendToLLM(string) подставляет системным сообщением
        // промпт КЛИЕНТА, а он у общего Settings.LLM не задан, поэтому поставщик отвергает content = null
        List<LLMMessage> messages = [new LLMMessage(LLMMessage.UserRole, promptForModel)];

        string json = await (_llm ?? Settings.LLM).SendToLLM(messages, _settings).ConfigureAwait(false);
        return JsonSerializer.Deserialize<Specifications>(json, JsonOptions) ?? new Specifications();
    }

    // Схема ответа: имена полей совпадают со свойствами Specifications,
    // список стилей берется из Style, чтобы не расходиться с перечислением
    private static string BuildSchema()
    {
        string styles = Names<Style>();
        string domains = Names<Domain>();
        string languages = Names<ProgrammingLanguage>();
        string sciences = Names<ScienceField>();
        string kinds = Names<TaskKind>();

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
                "readabilityScore": { "type": "number", "minimum": 0, "maximum": 100, "description": "Читаемость по Флешу-Кинкейду, 0-100" },
                "termDensity": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.TermDensity}}" },
                "formalityScore": { "type": "number", "minimum": 0, "maximum": 1, "description": "Формальность тона, 0-1" },
                "language": { "type": "string", "description": "Язык ответа, код ISO 639-1" },
                "hasReferences": { "type": "boolean", "description": "Нужны ли ссылки на источники" },
                "domain": { "type": "string", "enum": [{{domains}}], "description": "{{SpecFieldDescriptions.Domain}}" },
                "programmingLanguage": { "type": "string", "enum": [{{languages}}], "description": "{{SpecFieldDescriptions.ProgrammingLanguage}}" },
                "scienceField": { "type": "string", "enum": [{{sciences}}], "description": "{{SpecFieldDescriptions.ScienceField}}" },
                "taskKind": { "type": "string", "enum": [{{kinds}}], "description": "{{SpecFieldDescriptions.TaskKind}}" },
                "expertLevel": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.ExpertLevel}}" },
                "difficulty": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.Difficulty}}" },
                "factualityDemand": { "type": "number", "minimum": 0, "maximum": 1, "description": "{{SpecFieldDescriptions.FactualityDemand}}" },
                "requiredPoints": { "type": "array", "items": { "type": "string" }, "description": "{{SpecFieldDescriptions.RequiredPoints}}" },
                "constraints": { "type": "array", "items": { "type": "string" }, "description": "{{SpecFieldDescriptions.Constraints}}" }
              },
              "required": [
                "styleType", "symbolLength", "wordLength", "paragraphCount", "sectionCount",
                "listItemCount", "tableCount", "codeBlockCount", "formulaCount", "headingDepth",
                "avgSentenceLength", "readabilityScore", "termDensity", "formalityScore",
                "language", "hasReferences", "domain", "programmingLanguage", "scienceField", "taskKind",
                "expertLevel", "difficulty", "factualityDemand", "requiredPoints", "constraints"
              ],
              "additionalProperties": false
            }
            """;
    }

    // Список значений перечисления для схемы, в кавычках через запятую
    private static string Names<TEnum>() where TEnum : struct, Enum =>
        string.Join(", ", Enum.GetNames<TEnum>().Select(name => $"\"{name}\""));
}