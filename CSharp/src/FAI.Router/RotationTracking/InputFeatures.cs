using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;

namespace FAI.Router.RotationTracking;

/// <summary>
/// Признаки входа: объем запроса и ожидаемого ответа, свойства самой задачи и распознанное задание
/// </summary>
/// <remarks>
/// Свойства задачи (трудность, экспертность, число ограничений, опора на факты, длина диалога)
/// входят в вектор здесь, а не в вектор спецификации. У готового ответа их нет, и судья формы,
/// сравнивая заказ с фактом косинусом, видел бы в них расхождение там, где его нет.
/// </remarks>
public class InputFeatures
{
    // Типичные объемы в токенах. Счетчики входят в вектор признаков через логарифмическую
    // шкалу, как и объемы в спецификации. В сырых токенах они давали 100% длины вектора на
    // настоящем запросе, и все координаты спецификации весили ноль: роутер не видел типа задачи.
    private const double InputScale = 2000;
    private const double AnswerScale = 7000;
    private const double TurnScale = 10;
    private const double ConstraintScale = 10;

    public double InputLen { get; set; }
    public double LenAnswer { get; set; }

    /// <summary>
    /// Число реплик пользователя в диалоге. Одна реплика означает обычный запрос и в вектор ничего
    /// не добавляет; категория арены Multi-Turn различает модели как раз по длинным диалогам.
    /// </summary>
    public int TurnCount { get; set; } = 1;

    public Specifications InputSpecifications { get; set; } = new Specifications();

    /// <summary>
    /// Агрегация свойств в вектор
    /// </summary>
    public Vector FeatureVector
    {
        get
        {
            Vector features = new Vector(Settings.FeaturesDim);
            features[0] = Specifications.Scaled(InputLen, InputScale);
            features[1] = Specifications.Scaled(LenAnswer, AnswerScale);
            features[2] = Specifications.Scaled(TurnCount - 1, TurnScale);
            features[3] = Specifications.Scaled(InputSpecifications.Constraints.Count, ConstraintScale);
            features[4] = InputSpecifications.ExpertLevel;
            features[5] = InputSpecifications.Difficulty;
            features[6] = InputSpecifications.FactualityDemand;
            features.AddRange(InputSpecifications.FeaturesSpecificationVector);

            return features.GetUnitVector(); // Нормировка
        }
    }
}
