using FAI.Router.RotationTracking;

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

    /// <summary>
    /// Отдает признаки текста (промпта)
    /// </summary>
    /// <param name="text">Текст</param>
    public static InputFeatures GetFeatures(string text) 
    {
        InputFeatures features = new InputFeatures
        {
            InputLen = GetLenInput(text),
            LenAnswer = GetLenAnswer(text)
        };

        return features;
    }

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
}
