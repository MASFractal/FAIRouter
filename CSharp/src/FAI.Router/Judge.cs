using AI.DataStructs.Algebraic;

namespace FAI.Router;

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
    /// Фактическая спецификация выхода
    /// </summary>
    /// <param name="answer">Ответ</param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public Specifications GetSpec(string answer) => throw new NotImplementedException();
}

public class Specifications 
{
    public Vector FeaturesSpecificationVector { get; set; } = new Vector(Settings.FeaturesSpecDim)+1;
}
