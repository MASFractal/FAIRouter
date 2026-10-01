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
    /// <param name="content">Оценка содержания; с ней разбор включает и расхождения по содержанию</param>
    public static DiffSpec Criticize(Specifications inputSpec, Specifications actualSpec, ContentReview? content = null) =>
        DiffSpec.Compare(inputSpec, actualSpec, content);

    /// <summary>
    /// Итоговая оценка ответа: содержание и форма с долями из <see cref="Settings.ContentWeight"/>.
    /// Форма здесь это доля выполненных пунктов формы критика: число объяснимое и не зависящее от
    /// обучаемой матрицы. Без оценки содержания итог равен форме.
    /// </summary>
    /// <param name="critic">Разбор критика</param>
    /// <param name="content">Оценка содержания; пусто, если ее не делали</param>
    public static double Assess(DiffSpec critic, ContentReview? content)
    {
        double formScore = 1 - critic.FormDeviation;

        return content is null ? formScore : Settings.ContentWeight * content.Score + (1 - Settings.ContentWeight) * formScore;
    }

    /// <summary>
    /// Отчет: оценки содержания, формы и итог, затем все проваленные пункты ТЗ и замечания судьи
    /// </summary>
    /// <param name="critic">Разбор критика</param>
    /// <param name="content">Оценка содержания; пусто, если ее не делали</param>
    public static string Report(DiffSpec critic, ContentReview? content)
    {
        string form = Format(1 - critic.FormDeviation);
        string head = content is null
            ? $"Форма {form}"
            : $"Содержание {Format(content.Score)}, форма {form}, итог {Format(Assess(critic, content))}";

        IEnumerable<string> lines =
        [
            head,
            .. critic.Mismatches.Select(item => $"{item.Field}: заказано {item.Requested}, получено {item.Actual}"),
            .. (content?.Issues ?? []).Select(issue => "- " + issue),
        ];

        return string.Join(Environment.NewLine, lines);
    }

    private static string Format(double value) => value.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);
}
