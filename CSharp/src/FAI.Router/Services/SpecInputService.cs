using AI.LLM.Core.Models.Common.Requests;
using AI.LLM.Services.LLM;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.LLM;
using System.Text.Json;
using System.Text.Json.Serialization;
using static System.Net.Mime.MediaTypeNames;

namespace FAI.Router.Services;

public interface ISpecService 
{
    /// <summary>
    /// Получение спецификации
    /// </summary>
    /// <returns></returns>
    Task<Specifications> GetSpecificationsAsync(string text);
}

/// <summary>
/// Получение спецификации на входе (через LLM)
/// </summary>
public class SpecInputService : ISpecService
{
    
    private readonly LLMRecognitionSpecInput _specRecog;


    /// <summary>
    /// Получение спецификации на входе (через LLM)
    /// </summary>
    /// <param name="llm">Клиент модели; не задан, тогда берется общий Settings.LLM</param>
    public SpecInputService(LLMBase? llm = null)
    {
        _specRecog = new LLMRecognitionSpecInput(llm);
    }

    /// <summary>
    /// Извлекает ожидаемую спецификацию ответа из текста запроса (один запрос к LLM)
    /// </summary>
    /// <param name="prompt">Текст запроса пользователя</param>
    public async Task<Specifications> GetSpecificationsAsync(string prompt)
    {
        return await _specRecog.GetSpecificationsAsync(prompt);
    }

   
}

/// <summary>
/// Получение спецификации на выходе: структура считается по тексту, стиль распознается моделью
/// </summary>
public class SpecOutputService : ISpecService
{
    private readonly StyleClassifier _styleClassifier;

    /// <summary>
    /// Замер фактической спецификации ответа
    /// </summary>
    /// <param name="llm">Клиент модели; не задан, тогда берется общий Settings.LLM</param>
    public SpecOutputService(LLMBase? llm = null) => _styleClassifier = new StyleClassifier(llm);

    /// <summary>
    /// Измеряет фактическую спецификацию готового ответа
    /// </summary>
    /// <param name="answer">Текст ответа</param>
    public async Task<Specifications> GetSpecificationsAsync(string answer)
    {
        if (string.IsNullOrWhiteSpace(answer))
            throw new ArgumentException("Ответ не может быть пустым.", nameof(answer));

        Specifications specifications = TextMetrics.Measure(answer);
        StyleAssessment assessment = await _styleClassifier.AssessAsync(answer).ConfigureAwait(false);

        specifications.StyleType = assessment.StyleType;
        specifications.TermDensity = assessment.TermDensity;
        specifications.FormalityScore = assessment.FormalityScore;

        return specifications;
    }
}
