using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;

namespace FAI.Router.RotationTracking;

/// <summary>
/// Признаки входа
/// </summary>
public class InputFeatures 
{
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
            features[0] = InputLen;
            features[1] = LenAnswer;
            features.AddRange(InputSpecifications.FeaturesSpecificationVector);

            return features.GetUnitVector(); // Нормировка
        }
    }
}