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

    /// <summary>
    /// Итоговая оценка ответа: содержание и форма с долями из <see cref="Settings.ContentWeight"/>.
    /// Форма здесь это доля выполненных пунктов критика: число объяснимое и не зависящее от
    /// обучаемой матрицы. Без оценки содержания итог равен форме.
    /// </summary>
    /// <param name="form">Разбор формы</param>
    /// <param name="content">Оценка содержания; пусто, если ее не делали</param>
    public static double Assess(DiffSpec form, ContentReview? content)
    {
        double formScore = 1 - form.TotalDeviation;

        return content is null ? formScore : Settings.ContentWeight * content.Score + (1 - Settings.ContentWeight) * formScore;
    }

    /// <summary>
    /// Отчет по обеим осям: сначала содержание с замечаниями, потом проваленные пункты формы
    /// </summary>
    /// <param name="form">Разбор формы</param>
    /// <param name="content">Оценка содержания; пусто, если ее не делали</param>
    public static string Report(DiffSpec form, ContentReview? content)
    {
        string formPart = $"Форма {(1 - form.TotalDeviation).ToString("0.00", System.Globalization.CultureInfo.InvariantCulture)}"
            + (form.Mismatches.Any() ? Environment.NewLine + form : "");

        return content is null ? formPart : content + Environment.NewLine + formPart;
    }
}
