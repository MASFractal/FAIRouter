using AI.LLM.Core.Models.Common.Requests;
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
        Явно заданные требования (объём, стиль, число разделов, таблицы, язык) бери как есть.
        Неуказанное оценивай разумным ожиданием для такой задачи, а не нулём.

        Запрос пользователя:
        ----
        {text}
        ----
        """;

    private readonly GenerateSettings _settings;

    private static readonly string SchemaJson = BuildSchema();

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };

    /// <summary>
    /// Получение спецификации на входе (через LLM)
    /// </summary>
    public LLMRecognitionSpecInput()
    {
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

        string json = await Settings.LLM.SendToLLM(promptForModel, _settings).ConfigureAwait(false);
        return JsonSerializer.Deserialize<Specifications>(json, JsonOptions) ?? new Specifications();
    }

    // Схема ответа: имена полей совпадают со свойствами Specifications,
    // список стилей берётся из Style, чтобы не расходиться с перечислением
    private static string BuildSchema()
    {
        string styles = string.Join(", ", Enum.GetNames<Style>().Select(name => $"\"{name}\""));

        return $$"""
            {
              "type": "object",
              "properties": {
                "styleType": { "type": "string", "enum": [{{styles}}], "description": "Стиль текста ответа" },
                "symbolLength": { "type": "integer", "description": "Объём ответа в символах" },
                "wordLength": { "type": "integer", "description": "Объём ответа в словах" },
                "paragraphCount": { "type": "integer", "description": "Число абзацев" },
                "sectionCount": { "type": "integer", "description": "Число разделов" },
                "listItemCount": { "type": "integer", "description": "Число пунктов списков" },
                "tableCount": { "type": "integer", "description": "Число таблиц" },
                "codeBlockCount": { "type": "integer", "description": "Число блоков кода" },
                "formulaCount": { "type": "integer", "description": "Число формул" },
                "headingDepth": { "type": "integer", "description": "Глубина вложенности заголовков" },
                "avgSentenceLength": { "type": "number", "description": "Средняя длина предложения в словах" },
                "readabilityScore": { "type": "number", "description": "Читаемость по Флешу-Кинкейду, 0-100" },
                "termDensity": { "type": "number", "description": "Доля терминологии, 0-1" },
                "formalityScore": { "type": "number", "description": "Формальность тона, 0-1" },
                "language": { "type": "string", "description": "Язык ответа, код ISO 639-1" },
                "hasReferences": { "type": "boolean", "description": "Нужны ли ссылки на источники" }
              },
              "required": [
                "styleType", "symbolLength", "wordLength", "paragraphCount", "sectionCount",
                "listItemCount", "tableCount", "codeBlockCount", "formulaCount", "headingDepth",
                "avgSentenceLength", "readabilityScore", "termDensity", "formalityScore",
                "language", "hasReferences"
              ],
              "additionalProperties": false
            }
            """;
    }
}