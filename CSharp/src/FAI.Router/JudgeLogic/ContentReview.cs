using System.Globalization;
using System.Text;

namespace FAI.Router.JudgeLogic;

/// <summary>
/// Проверяемое утверждение ответа и вероятность, что оно верно
/// </summary>
/// <param name="Text">Утверждение одной фразой</param>
/// <param name="Truth">Вероятность истинности, 0-1</param>
public sealed record FactClaim(string Text, double Truth);

/// <summary>
/// Оценка одного критерия содержания
/// </summary>
/// <param name="Name">Название критерия</param>
/// <param name="Score">Оценка 0-1; пусто, если к этой задаче критерий не относится</param>
public sealed record ContentCriterion(string Name, double? Score);

/// <summary>
/// Оценка содержания ответа: то, что не видно по форме. Таблица с выдуманными цифрами,
/// отчет не про ту аналитику и несуществующие источники проходят сверку формы на отлично, а
/// здесь проваливаются.
/// </summary>
/// <remarks>
/// Критерий, который к задаче не относится, в среднее не входит: у задачи без ограничений нечего
/// выполнять, у ответа без проверяемых утверждений нечего проверять. Воздержание от фактов не
/// штрафуется, ложный факт штрафуется, как в рейтинге фактологии арены.
/// </remarks>
public sealed class ContentReview
{
    public const string Factuality = "Фактология";
    public const string Completeness = "Полнота по сути";
    public const string InstructionFollowing = "Выполнение указаний";
    public const string Reasoning = "Верность рассуждений и расчетов";
    public const string Expertise = "Экспертная глубина";
    public const string StructureContent = "Наполнение структуры";
    public const string SourceQuality = "Качество источников";
    public const string FitForPurpose = "Пригодность для дела";

    /// <summary>Оценка, ниже которой критерий считается проваленным</summary>
    public const double PassMark = 0.6;

    /// <summary>Критерии по порядку</summary>
    public IReadOnlyList<ContentCriterion> Criteria { get; }

    /// <summary>Проверяемые утверждения ответа</summary>
    public IReadOnlyList<FactClaim> Claims { get; }

    /// <summary>Конкретные замечания по содержанию: что неверно или упущено и где</summary>
    public IReadOnlyList<string> Issues { get; }

    /// <summary>Оценка содержания: среднее по критериям, которые относятся к задаче</summary>
    public double Score =>
        Criteria.Where(item => item.Score is not null).Select(item => item.Score!.Value).DefaultIfEmpty(1).Average();

    public ContentReview(IReadOnlyList<ContentCriterion> criteria, IReadOnlyList<FactClaim> claims, IReadOnlyList<string> issues)
    {
        Criteria = criteria;
        Claims = claims;
        Issues = issues;
    }

    /// <summary>Оценка критерия по названию; пусто, если критерий к задаче не относится</summary>
    public double? Get(string name) => Criteria.FirstOrDefault(item => item.Name == name)?.Score;

    /// <summary>
    /// Фактология по утверждениям: средняя вероятность истинности; утверждений нет, значит пусто
    /// </summary>
    public static double? FactualityOf(IReadOnlyCollection<FactClaim> claims) =>
        claims.Count == 0 ? null : claims.Average(claim => Math.Clamp(claim.Truth, 0, 1));

    /// <summary>
    /// Отчет: оценка содержания, проваленные критерии, ложные утверждения и замечания
    /// </summary>
    public override string ToString()
    {
        StringBuilder report = new();
        report.AppendLine($"Содержание {Format(Score)}");

        foreach (ContentCriterion criterion in Criteria.Where(item => item.Score < PassMark))
            report.AppendLine($"  {criterion.Name}: {Format(criterion.Score!.Value)}");

        foreach (FactClaim claim in Claims.Where(item => item.Truth < 0.5))
            report.AppendLine($"  Сомнительное утверждение ({Format(claim.Truth)}): {claim.Text}");

        foreach (string issue in Issues)
            report.AppendLine($"  - {issue}");

        return report.ToString().TrimEnd();
    }

    private static string Format(double value) => value.ToString("0.00", CultureInfo.InvariantCulture);
}
