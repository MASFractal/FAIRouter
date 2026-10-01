namespace FAI.Router.JudgeLogic;

/// <summary>
/// Расхождение по одному пункту ТЗ
/// </summary>
/// <param name="Field">Название пункта</param>
/// <param name="Requested">Заказано</param>
/// <param name="Actual">Получено</param>
/// <param name="Deviation">Отклонение: ноль означает совпадение, единица означает полное расхождение</param>
/// <param name="Content">Пункт содержания; ложь означает пункт формы</param>
public record SpecDeviation(string Field, string Requested, string Actual, double Deviation, bool Content = false);

/// <summary>
/// Разбор всех расхождений между ТЗ и фактом по каждому пункту (режим критика)
/// </summary>
/// <remarks>
/// Форма сверяется всегда: 20 пунктов от стиля и объема до типа задачи. С оценкой содержания
/// разбор включает и его: каждый смысловой пункт заказа, каждое ограничение, экспертность ответа
/// против заказанной, каждое проверяемое утверждение, расчеты, наполнение структуры, источники и
/// пригодность для дела. Трудности и длины диалога в разборе нет: это свойства запроса, у ответа
/// им нечего противопоставить.
/// </remarks>
public class DiffSpec
{
    /// <summary>
    /// Отклонение, начиная с которого пункт считается проваленным
    /// </summary>
    public const double MismatchThreshold = 0.2;

    /// <summary>
    /// Расхождения по всем пунктам ТЗ
    /// </summary>
    public IReadOnlyList<SpecDeviation> Deviations { get; }

    /// <summary>
    /// Среднее отклонение по всем пунктам, формы и содержания
    /// </summary>
    public double TotalDeviation => Deviations.Average(deviation => deviation.Deviation);

    /// <summary>
    /// Среднее отклонение по пунктам формы: итог формы равен единице минус это число
    /// </summary>
    public double FormDeviation => Deviations.Where(deviation => !deviation.Content).Average(deviation => deviation.Deviation);

    /// <summary>
    /// Проваленные пункты, худшие первыми
    /// </summary>
    public IEnumerable<SpecDeviation> Mismatches =>
        Deviations.Where(deviation => deviation.Deviation > MismatchThreshold)
                  .OrderByDescending(deviation => deviation.Deviation);

    private DiffSpec(IReadOnlyList<SpecDeviation> deviations) => Deviations = deviations;

    /// <summary>
    /// Сравнивает заказанную и фактическую спецификации по каждому пункту
    /// </summary>
    /// <param name="requested">Заказано (ТЗ)</param>
    /// <param name="actual">Получено по факту</param>
    /// <param name="content">Оценка содержания; с ней разбор включает расхождения по содержанию</param>
    public static DiffSpec Compare(Specifications requested, Specifications actual, ContentReview? content = null) =>
        new([.. Form(requested, actual), .. content is null ? [] : Content(requested, content)]);

    /// <summary>
    /// Отчет критика: список проваленных пунктов
    /// </summary>
    public override string ToString() =>
        string.Join(Environment.NewLine,
            Mismatches.Select(item => $"{item.Field}: заказано {item.Requested}, получено {item.Actual}"));

    private static SpecDeviation[] Form(Specifications requested, Specifications actual) =>
    [
        Exact("Стиль", requested.StyleType, actual.StyleType),
        Number("Объем в символах", requested.SymbolLength, actual.SymbolLength),
        Number("Объем в словах", requested.WordLength, actual.WordLength),
        Number("Абзацы", requested.ParagraphCount, actual.ParagraphCount),
        Number("Разделы", requested.SectionCount, actual.SectionCount),
        Number("Пункты списков", requested.ListItemCount, actual.ListItemCount),
        Number("Таблицы", requested.TableCount, actual.TableCount),
        Number("Блоки кода", requested.CodeBlockCount, actual.CodeBlockCount),
        Number("Формулы", requested.FormulaCount, actual.FormulaCount),
        Number("Глубина заголовков", requested.HeadingDepth, actual.HeadingDepth),
        Number("Средняя длина предложения", requested.AvgSentenceLength, actual.AvgSentenceLength),
        Number("Читаемость", requested.ReadabilityScore, actual.ReadabilityScore),
        Number("Доля терминологии", requested.TermDensity, actual.TermDensity),
        Number("Формальность", requested.FormalityScore, actual.FormalityScore),
        Exact("Язык", requested.Language, actual.Language),
        Exact("Ссылки на источники", requested.HasReferences, actual.HasReferences),
        Exact("Область", requested.Domain, actual.Domain),
        Exact("Язык программирования", requested.ProgrammingLanguage, actual.ProgrammingLanguage),
        Exact("Область науки", requested.ScienceField, actual.ScienceField),
        Exact("Тип задачи", requested.TaskKind, actual.TaskKind)
    ];

    // Пункты содержания: у каждого смыслового пункта и ограничения своя строка, у каждого
    // проверяемого утверждения тоже. Общая оценка критерия идет строкой, только когда поштучного
    // разбора нет: иначе одно расхождение попало бы в разбор дважды
    private static IEnumerable<SpecDeviation> Content(Specifications requested, ContentReview content)
    {
        foreach (PointCoverage point in content.Points)
            yield return new($"Смысловой пункт «{point.Point}»", "раскрыть", CoverageText(point.Coverage), 1 - point.Coverage, true);

        if (content.Points.Count == 0)
            yield return Criterion(content, ContentReview.Completeness);

        foreach (ConstraintCheck check in content.ConstraintChecks)
            yield return new($"Ограничение «{check.Constraint}»", "соблюсти", check.Met ? "соблюдено" : "нарушено", check.Met ? 0 : 1, true);

        if (requested.ExpertLevel > 0 && content.ExpertLevel is { } level)
            yield return Number("Экспертность", requested.ExpertLevel, level) with { Content = true };
        else
            yield return Criterion(content, ContentReview.Expertise);

        foreach (FactClaim claim in content.Claims)
            yield return new($"Факт «{claim.Text}»", "верно", $"верно с вероятностью {claim.Truth:0.00}", 1 - claim.Truth, true);

        foreach (string name in (string[])[ContentReview.Reasoning, ContentReview.StructureContent, ContentReview.SourceQuality, ContentReview.FitForPurpose])
            if (content.Get(name) is not null)
                yield return Criterion(content, name);
    }

    private static SpecDeviation Criterion(ContentReview content, string name)
    {
        double score = content.Get(name) ?? 1;

        return new(name, "1", score.ToString("G4"), 1 - score, true);
    }

    private static string CoverageText(double coverage) =>
        coverage >= 0.8 ? "раскрыт" : coverage >= 0.3 ? "раскрыт частично" : "не раскрыт";

    // Числовой пункт: относительное отклонение от заказанного
    private static SpecDeviation Number(string field, double requested, double actual)
    {
        // Заказан ноль, а получено больше нуля: расхождение полное, потому что делить не на что
        double deviation = requested == 0
            ? (actual == 0 ? 0 : 1)
            : Math.Min(1, Math.Abs(actual - requested) / Math.Abs(requested));

        return new SpecDeviation(field, requested.ToString("G4"), actual.ToString("G4"), deviation);
    }

    // Категориальный пункт: совпало или нет
    private static SpecDeviation Exact(string field, object? requested, object? actual) =>
        new(field, requested?.ToString() ?? "нет", actual?.ToString() ?? "нет", Equals(requested, actual) ? 0 : 1);
}
