using AI.LLM.Services.LLM;
using FAI.Router.JudgeLogic;
using FAI.Router.LLM;

namespace FAI.Router.Services;

/// <summary>
/// Получение спецификации текста: заказа по запросу или факта по ответу
/// </summary>
public interface ISpecService
{
    /// <summary>
    /// Получение спецификации
    /// </summary>
    /// <param name="text">Текст запроса или ответа</param>
    /// <param name="cancellationToken">Токен отмены</param>
    Task<Specifications> GetSpecificationsAsync(string text, CancellationToken cancellationToken = default);
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
    public SpecInputService(LLMBase? llm = null) => _specRecog = new LLMRecognitionSpecInput(llm);

    /// <summary>
    /// Извлекает ожидаемую спецификацию ответа из текста запроса (один запрос к LLM)
    /// </summary>
    /// <param name="prompt">Текст запроса пользователя</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public Task<Specifications> GetSpecificationsAsync(string prompt, CancellationToken cancellationToken = default) =>
        _specRecog.GetSpecificationsAsync(prompt, cancellationToken);
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
    /// <param name="cancellationToken">Токен отмены</param>
    public async Task<Specifications> GetSpecificationsAsync(string answer, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(answer))
            throw new ArgumentException("Ответ не может быть пустым.", nameof(answer));

        Specifications specifications = TextMetrics.Measure(answer);
        StyleAssessment assessment = await _styleClassifier.AssessAsync(answer, cancellationToken).ConfigureAwait(false);

        specifications.StyleType = assessment.StyleType;
        specifications.TermDensity = assessment.TermDensity;
        specifications.FormalityScore = assessment.FormalityScore;
        specifications.Domain = assessment.Domain;
        specifications.ScienceField = assessment.ScienceField;
        specifications.TaskKind = assessment.TaskKind;

        return specifications;
    }
}
