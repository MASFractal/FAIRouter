using AI.DataStructs.Algebraic;

namespace FAI.Router;

/// <summary>
/// Базовый элемент для роутинга
/// Соответствует модели
/// </summary>
public class BaseRoutedElement
{
    /// <summary>
    /// Имя элемента
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Вектор для сравнения 
    /// </summary>
    public Vector IdealMatchVector { get; set; } = new Vector(Settings.FeaturesDim) + 1.0/Settings.FeaturesDim; // Простое среднее признаков

    /// <summary>
    /// Число токенов в секунду
    /// </summary>
    public double TPS { get; set; } = 1;

    /// <summary>
    /// Цена долларов за 1 млн токенов (вход)
    /// </summary>
    public double DPMTInp { get; set; }

    /// <summary>
    /// Цена долларов за 1 млн токенов (выход)
    /// </summary>
    public double DPMTOutp { get; set; }

    /// <summary>
    /// Получение оценки качества для данного элемента роутинга
    /// по умолчанию скалярное произведение
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public virtual double GetQualityScore(Vector features) =>
        features.Dot(IdealMatchVector);

    /// <summary>
    /// Оценка по метрике R
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public double GetRScore(InputFeatures features)
    {
        double t = features.LenAnswer / TPS; // time
        double c = (DPMTInp * features.InputLen + DPMTOutp * features.LenAnswer) * 1e-6; // cost
        double q = GetQualityScore(features.FeatureVector);
        double nom = Settings.WQ * q - Settings.WC * c;
        double denom = Settings.Wt * Math.Log(t + 1);
        return nom / denom;
    }
}
