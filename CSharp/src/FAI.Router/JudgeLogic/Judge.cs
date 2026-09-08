using AI.DataStructs.Algebraic;
using FAI.Router.RotationTracking;

namespace FAI.Router.JudgeLogic;


/// <summary>
/// Судья, автоматическая оценка качества решения.
/// Замером факта судья не занимается: это работа SpecOutputService, и обе спецификации
/// приходят к судье готовыми. Иначе зона оценки зависела бы от зоны сервисов, а та уже
/// зависит от нее, из-за чего получилась бы кольцевая связь.
/// </summary>
public class Judge
{
    /// <summary>
    /// Матрица трансформации (обучаемая)
    /// </summary>
    public Matrix TransformerW { get; set; } = Matrix.Identity(Settings.FeaturesSpecDim);


    /// <summary>
    /// Оценка качества: близость факта к заказу, пропущенному через обучаемую матрицу
    /// </summary>
    /// <param name="inputSpec">Запрашиваемые параметры (ТЗ)</param>
    /// <param name="actualSpec">Фактические параметры ответа</param>
    public double GetScore(Specifications inputSpec, Specifications actualSpec)
    {
        Vector transformed = TransformerW.MulMatrOnVectColumn(inputSpec.FeaturesSpecificationVector);

        return actualSpec.FeaturesSpecificationVector.Cos(transformed);
    }

    /// <summary>
    /// Проставляет оценку хода в его трассировку и тем замыкает круг: ход состоялся,
    /// судья его оценил, и трассировка стала обучающим примером
    /// </summary>
    /// <param name="trace">Трассировка хода</param>
    /// <param name="inputSpec">Запрашиваемые параметры (ТЗ)</param>
    /// <param name="actualSpec">Фактические параметры ответа</param>
    public double Rate(Tracert trace, Specifications inputSpec, Specifications actualSpec)
    {
        trace.Score = GetScore(inputSpec, actualSpec);

        return trace.Score;
    }

    /// <summary>
    /// Режим критика: расхождения между ТЗ и фактом по каждому пункту.
    /// В отличие от GetScore сравнивает пункты по отдельности, без вектора и матрицы,
    /// поэтому объяснимо для человека и не зависит от масштаба координат.
    /// </summary>
    /// <param name="inputSpec">Запрашиваемые параметры</param>
    /// <param name="actualSpec">Фактические параметры ответа</param>
    public static DiffSpec Criticize(Specifications inputSpec, Specifications actualSpec) =>
        DiffSpec.Compare(inputSpec, actualSpec);
}
