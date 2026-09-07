using AI.DataStructs.Algebraic;

namespace FAI.Router.JudgeLogic;


/// <summary>
/// Судья, автоматическая оценка качества решения
/// </summary>
public class Judge
{
    /// <summary>
    /// Матрица трансформации (обучаемая)
    /// </summary>
    public Matrix TransformerW { get; set; } = Matrix.Identity(Settings.FeaturesSpecDim);


    /// <summary>
    /// Оценка качества
    /// </summary>
    /// <param name="inputSpec">Запрашиваемые параметры</param>
    /// <param name="answer">Фактический ответ</param>
    /// <returns></returns>
    public double GetScore(Specifications inputSpec, string answer) 
    {
        var spec = GetSpec(answer);
        var trVect = TransformerW.MulMatrOnVectColumn(
            inputSpec.FeaturesSpecificationVector);
        return spec.FeaturesSpecificationVector.Cos(trVect);
    }

    /// <summary>
    /// Режим критика: расхождения между ТЗ и фактом по каждому пункту.
    /// В отличие от GetScore сравнивает пункты по отдельности, без вектора и матрицы —
    /// поэтому объяснимо для человека и не зависит от масштаба координат.
    /// </summary>
    /// <param name="inputSpec">Запрашиваемые параметры</param>
    /// <param name="actualSpec">Фактические параметры ответа</param>
    public static DiffSpec Criticize(Specifications inputSpec, Specifications actualSpec) =>
        DiffSpec.Compare(inputSpec, actualSpec);

    /// <summary>
    /// Фактическая спецификация выхода
    /// </summary>
    /// <param name="answer">Ответ</param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public Specifications GetSpec(string answer) => throw new NotImplementedException();
}
