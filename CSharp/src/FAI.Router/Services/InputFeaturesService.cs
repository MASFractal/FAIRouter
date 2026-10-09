using FAI.Router.JudgeLogic;
using FAI.Router.RotationTracking;
using FAI.Router.Training;

namespace FAI.Router.Services;

/// <summary>
/// Сервис вычисления признаков входа
/// </summary>
public class InputFeaturesService
{
    /// <summary>
    /// Символов на токен
    /// </summary>
    public const double EST_SYMBOL_PER_TOKEN = 3.0;

    private static readonly SpecInputService SpecService = new();

    /// <summary>
    /// Полные признаки запроса: объем оценивается арифметикой, ТЗ распознает модель. Сбой
    /// распознавания хода не роняет: признаки остаются как у <see cref="GetFeatures"/>.
    /// </summary>
    /// <param name="text">Текст запроса</param>
    /// <param name="specs">Кто распознает задание; пусто, значит общий распознаватель через Settings.LLM</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public static async Task<InputFeatures> GetFeaturesAsync(string text, ISpecService? specs = null, CancellationToken cancellationToken = default)
    {
        InputFeatures features = GetFeatures(text);

        if (await RecognizeAsync(text, specs, cancellationToken).ConfigureAwait(false) is { } recognized)
            Apply(features, recognized);

        return features;
    }

    /// <summary>
    /// Распознанное задание; пусто, если распознать не удалось
    /// </summary>
    /// <remarks>
    /// Ловится все, кроме отмены вызывающим: обрезанный или негодный ответ модели, сеть, отказ
    /// поставщика, исчерпанный бюджет. Прежде такой сбой ронял ход до выбора исполнителя. В журнал
    /// пишется только тип ошибки: текст исключения движка несет начало запроса пользователя.
    /// </remarks>
    /// <param name="text">Текст запроса</param>
    /// <param name="specs">Кто распознает задание; пусто, значит общий распознаватель через Settings.LLM</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public static async Task<Specifications?> RecognizeAsync(string text, ISpecService? specs = null, CancellationToken cancellationToken = default)
    {
        try
        {
            return await (specs ?? SpecService).GetSpecificationsAsync(text, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception error) when (!cancellationToken.IsCancellationRequested)
        {
            System.Diagnostics.Trace.TraceWarning($"FAIRouter: задание не распознано ({error.GetType().Name}), выбор идет по типовой задаче.");
            return null;
        }
    }

    /// <summary>
    /// Признаки без обращения к модели: объем по длине текста, задание как у типовой задачи
    /// </summary>
    /// <remarks>
    /// Задание без распознавания неизвестно, а пустая спецификация это не «неизвестно», а крайняя
    /// точка: стиль «нет», читаемость и формальность ноль. Прогноз качества проецировался на это
    /// случайное направление, и порядок кандидатов не совпадал даже с их общей силой. Замер «выбрось
    /// серию рейтингов и предскажи ее» (docs/research/prior-holdout.md): лидер угадан в 3,7 % серий
    /// против 27,2 % со спецификацией типовой задачи и 34,6 % с полным распознаванием.
    /// </remarks>
    /// <param name="text">Текст</param>
    public static InputFeatures GetFeatures(string text) => new()
    {
        InputLen = GetLenInput(text),
        LenAnswer = GetLenAnswer(text),
        InputSpecifications = BenchmarkPrior.TypicalTask().InputSpecifications,
    };

    /// <summary>
    /// Получить оценку длинны
    /// </summary>
    /// <param name="text">Входной текст</param>
    public static double GetLenAnswer(string text) => 2*text.Length / EST_SYMBOL_PER_TOKEN; // Заказанный или средний объем

    /// <summary>
    /// Оценка длины входного текста в токенах
    /// </summary>
    /// <param name="text">Текст</param>
    /// <returns></returns>
    public static double GetLenInput(string text) => text.Length / EST_SYMBOL_PER_TOKEN;

    /// <summary>
    /// Ставит распознанное задание в признаки. Объем ответа берется из заказа; догадка по длине
    /// промпта остается на случай, когда заказ объема не назвал. Раньше цена и время считались только
    /// по промпту: «напиши обзор на двадцать тысяч знаков» это короткий запрос, и ход выглядел дешевым
    /// и быстрым у всех кандидатов разом.
    /// </summary>
    internal static void Apply(InputFeatures features, Specifications recognized)
    {
        features.InputSpecifications = recognized;

        if (recognized.SymbolLength > 0)
            features.LenAnswer = recognized.SymbolLength / EST_SYMBOL_PER_TOKEN;
    }
}
