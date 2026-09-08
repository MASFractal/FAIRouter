using AI.DataStructs.Algebraic;
using FAI.Router.RotationTracking;

namespace FAI.Router.RoutedElements;

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
    /// Вектор для сравнения (обучаемый), инициализация по Ксавье
    /// </summary>
    public Vector IdealMatchVector { get; set; } = XavierVector(Settings.FeaturesDim + Settings.FeaturesSpecDim);

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
        Settings.Center(features).Dot(IdealMatchVector);

    /// <summary>
    /// Оценка по метрике R
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public double GetRScore(InputFeatures features)
    {
        double t = features.LenAnswer / (TPS+0.1); // time
        double c = (DPMTInp * features.InputLen + DPMTOutp * features.LenAnswer) * 1e-6; // cost
        double q = GetQualityScore(features.FeatureVector);
        double nom = Settings.WQ * q - Settings.WC * c;
        double denom = Settings.Wt * Math.Log(t + 2);
        return nom / denom;
    }

    /// <summary>
    /// Инициализация обучаемого вектора по Ксавье: равномерно из [-limit, limit],
    /// limit = sqrt(6 / n). Разброс задан размерностью, поэтому прогноз качества на старте
    /// не зависит от того, сколько признаков в векторе.
    /// </summary>
    /// <param name="dimension">Размерность вектора признаков</param>
    public static Vector XavierVector(int dimension)
    {
        double limit = Math.Sqrt(6.0 / dimension);
        Vector vector = new(dimension);

        for (int i = 0; i < dimension; i++)
            vector[i] = (Random.Shared.NextDouble() * 2 - 1) * limit;

        return vector;
    }
}
