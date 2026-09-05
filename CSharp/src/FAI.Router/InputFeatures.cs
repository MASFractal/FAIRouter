using AI.DataStructs.Algebraic;

namespace FAI.Router;

/// <summary>
/// Признаки входа
/// </summary>
public class InputFeatures 
{
    public double InputLen { get; set; }
    public double LenAnswer { get; set; }

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
            return features.GetUnitVector(); // Нормировка
        }
    }
}