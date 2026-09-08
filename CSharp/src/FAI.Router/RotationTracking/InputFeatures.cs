using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;

namespace FAI.Router.RotationTracking;

/// <summary>
/// Признаки входа
/// </summary>
public class InputFeatures
{
    // Типичные объемы в токенах. Счетчики входят в вектор признаков через логарифмическую
    // шкалу, как и объемы в спецификации. В сырых токенах они давали 100% длины вектора на
    // настоящем запросе, и все координаты спецификации весили ноль: роутер не видел типа задачи.
    private const double InputScale = 2000;
    private const double AnswerScale = 7000;

    public double InputLen { get; set; }
    public double LenAnswer { get; set; }

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
            features.AddRange(InputSpecifications.FeaturesSpecificationVector);

            return features.GetUnitVector(); // Нормировка
        }
    }
}
