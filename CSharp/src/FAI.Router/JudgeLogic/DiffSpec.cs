using System.Globalization;
using FAI.Router.Enums;

namespace FAI.Router.JudgeLogic;

/// <summary>
/// Расхождение по одному пункту ТЗ
/// </summary>
/// <param name="Field">Название пункта</param>
/// <param name="Requested">Заказано</param>
/// <param name="Actual">Получено</param>
/// <param name="Deviation">Отклонение: ноль означает совпадение, единица означает полное расхождение</param>
/// <param name="Content">Пункт содержания; ложь означает пункт формы</param>
public record SpecDeviation(string Field, string Requested, string Actual, double Deviation, bool Content = false)
{
    /// <summary>Строка отчета: что заказано и что получено</summary>
    public override string ToString() => $"{Field}: заказано {Requested}, получено {Actual}";
}

/// <summary>
/// Разбор всех расхождений между ТЗ и фактом по каждому пункту (режим критика)
/// </summary>
/// <remarks>
/// Форма сверяется по полям, которые запрос задал явно (<see cref="Specifications.ExplicitFields"/>):
/// угаданное моделью «разумное ожидание» требованием не считается. Предмет задачи (область, область
/// науки, тип задачи) сверяется всегда. Объем идет одной строкой: символы и слова почти коллинеарны,
/// и две строки удваивали его вес. Счетчики сверяются относительным отклонением, шкалы с границами
/// (читаемость, доля терминологии, формальность, экспертность) разностью в долях размаха: на них
/// относительное отклонение штрафовало ответ лучше заказа сильнее провала. Незаказанные источники
/// не штрафуются, блок кода без подписи языка расхождением не считается, язык неизвестен у
/// замера, тогда не сверяется. С оценкой содержания разбор включает и его: каждый смысловой пункт
/// заказа, каждое ограничение, экспертность ответа против заказанной, каждое проверяемое
/// утверждение, расчеты, наполнение структуры, источники и пригодность для дела.
/// </remarks>
public class DiffSpec
{
    /// <summary>
    /// Отклонение, начиная с которого пункт считается проваленным
    /// </summary>
    public const double MismatchThreshold = 0.2;

    /// <summary>
    /// Оценка, ниже которой критерий, пункт или утверждение считаются проваленными: один порог на
    /// критика и на отчет содержания
    /// </summary>
    public const double PassMark = 1 - MismatchThreshold;

    /// <summary>
    /// Расхождения по всем пунктам ТЗ
    /// </summary>
    public IReadOnlyList<SpecDeviation> Deviations { get; }

    /// <summary>
    /// Среднее отклонение по всем пунктам, формы и содержания; ноль, если сверять нечего
    /// </summary>
    public double TotalDeviation => Deviations.Select(deviation => deviation.Deviation).DefaultIfEmpty(0).Average();

    /// <summary>
    /// Среднее отклонение по пунктам формы: итог формы равен единице минус это число
    /// </summary>
    public double FormDeviation =>
        Deviations.Where(deviation => !deviation.Content).Select(deviation => deviation.Deviation).DefaultIfEmpty(0).Average();

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
    public override string ToString() => string.Join(Environment.NewLine, Mismatches);

    private static IEnumerable<SpecDeviation> Form(Specifications requested, Specifications actual)
    {
        SpecDeviation?[] rows =
        [
            Stated(requested, "styleType", Exact("Стиль", requested.StyleType, actual.StyleType)),
            Volume(requested, actual),
            Stated(requested, "paragraphCount", Number("Абзацы", requested.ParagraphCount, actual.ParagraphCount)),
            Stated(requested, "sectionCount", Number("Разделы", requested.SectionCount, actual.SectionCount)),
            Stated(requested, "listItemCount", Number("Пункты списков", requested.ListItemCount, actual.ListItemCount)),
            Stated(requested, "tableCount", Number("Таблицы", requested.TableCount, actual.TableCount)),
            Stated(requested, "codeBlockCount", Number("Блоки кода", requested.CodeBlockCount, actual.CodeBlockCount)),
            Stated(requested, "formulaCount", Number("Формулы", requested.FormulaCount, actual.FormulaCount)),
            Stated(requested, "headingDepth", Number("Глубина заголовков", requested.HeadingDepth, actual.HeadingDepth)),
            Stated(requested, "avgSentenceLength", Number("Средняя длина предложения", requested.AvgSentenceLength, actual.AvgSentenceLength)),
            Stated(requested, "readabilityScore", Scale("Читаемость", requested.ReadabilityScore, actual.ReadabilityScore, 100)),
            Stated(requested, "termDensity", Scale("Доля терминологии", requested.TermDensity, actual.TermDensity, 1)),
            Stated(requested, "formalityScore", Scale("Формальность", requested.FormalityScore, actual.FormalityScore, 1)),
            Stated(requested, "language", Language(requested, actual)),
            Stated(requested, "hasReferences", References(requested, actual)),
            Stated(requested, "programmingLanguage", Code(requested, actual)),
            Exact("Область", requested.Domain, actual.Domain),
            Exact("Область науки", requested.ScienceField, actual.ScienceField),
            Exact("Тип задачи", requested.TaskKind, actual.TaskKind)
        ];

        return rows.OfType<SpecDeviation>();
    }

    // Пункты содержания: у каждого смыслового пункта и ограничения своя строка, у каждого
    // проверяемого утверждения тоже. Общая оценка критерия идет строкой, только когда поштучного
    // разбора нет: иначе одно расхождение попало бы в разбор дважды
    private static IEnumerable<SpecDeviation> Content(Specifications requested, ContentReview content)
    {
        foreach (PointCoverage point in content.Points)
            yield return new($"Смысловой пункт «{point.Point}»", "раскрыть", CoverageText(point.Coverage), 1 - point.Coverage, true);

        if (content.Points.Count == 0 && Criterion(content, ContentReview.Completeness) is { } completeness)
            yield return completeness;

        foreach (ConstraintCheck check in content.ConstraintChecks)
            yield return new($"Ограничение «{check.Constraint}»", "соблюсти", check.Met ? "соблюдено" : "нарушено", check.Met ? 0 : 1, true);

        if (requested.ExpertLevel > 0 && content.ExpertLevel is { } level)
            yield return Scale("Экспертность", requested.ExpertLevel, level, 1) with { Content = true };
        else if (Criterion(content, ContentReview.Expertise) is { } expertise)
            yield return expertise;

        foreach (FactClaim claim in content.Claims)
            yield return new($"Факт «{claim.Text}»", "верно", $"верно с вероятностью {Text(claim.Truth, "0.00")}", 1 - claim.Truth, true);

        foreach (string name in (string[])[ContentReview.Reasoning, ContentReview.StructureContent, ContentReview.SourceQuality, ContentReview.FitForPurpose])
            if (Criterion(content, name) is { } row)
                yield return row;
    }

    // Строка критерия содержания; пусто, если критерий к задаче не относится
    private static SpecDeviation? Criterion(ContentReview content, string name) =>
        content.Get(name) is { } score ? new(name, "1", Text(score, "G4"), 1 - score, true) : null;

    private static string CoverageText(double coverage) =>
        coverage >= PassMark ? "раскрыт" : coverage >= 0.3 ? "раскрыт частично" : "не раскрыт";

    // Строка поля формы, если запрос задал его явно; угаданное не сверяется
    private static SpecDeviation? Stated(Specifications requested, string field, SpecDeviation? row) =>
        requested.IsExplicit(field) ? row : null;

    // Объем одной строкой: та мера, что задана явно. Заданы обе или список неизвестен, тогда
    // символы, если они заказаны, иначе слова
    private static SpecDeviation? Volume(Specifications requested, Specifications actual)
    {
        bool symbols = requested.IsExplicit("symbolLength");
        bool words = requested.IsExplicit("wordLength");

        if (symbols && (!words || requested.SymbolLength > 0 || requested.WordLength == 0))
            return Number("Объем в символах", requested.SymbolLength, actual.SymbolLength);

        return words ? Number("Объем в словах", requested.WordLength, actual.WordLength) : null;
    }

    // Язык сравнивается кодом без региона и регистра; неизвестен с любой стороны, тогда не сверяется
    private static SpecDeviation? Language(Specifications requested, Specifications actual)
    {
        string? want = Specifications.NormalizeLanguage(requested.Language);
        string? got = Specifications.NormalizeLanguage(actual.Language);

        return want is null || got is null ? null : new("Язык", want, got, want == got ? 0 : 1);
    }

    // Незаказанные источники не штрафуются: штраф за них только при явном запрете, а явность
    // известна лишь из списка явных полей
    private static SpecDeviation References(Specifications requested, Specifications actual)
    {
        bool forbidden = requested.ExplicitFields is not null;
        double deviation = requested.HasReferences == actual.HasReferences ? 0 : requested.HasReferences || forbidden ? 1 : 0;

        return new("Ссылки на источники", Yes(requested.HasReferences), Yes(actual.HasReferences), deviation);
    }

    // Код есть, а языка замер не знает (блоки без подписи): это не расхождение
    private static SpecDeviation? Code(Specifications requested, Specifications actual) =>
        actual.ProgrammingLanguage == ProgrammingLanguage.None && actual.CodeBlockCount > 0
            ? null
            : Exact("Язык программирования", requested.ProgrammingLanguage, actual.ProgrammingLanguage);

    // Счетчик: относительное отклонение от заказанного. Заказан ноль, а получено больше нуля:
    // расхождение полное, потому что делить не на что; явный ноль это запрет
    private static SpecDeviation Number(string field, double requested, double actual)
    {
        double deviation = requested == 0
            ? (actual == 0 ? 0 : 1)
            : Math.Min(1, Math.Abs(actual - requested) / Math.Abs(requested));

        return new SpecDeviation(field, Text(requested, "G4"), Text(actual, "G4"), deviation);
    }

    // Шкала с границами: разность в долях размаха, одинаково в обе стороны от заказа
    private static SpecDeviation Scale(string field, double requested, double actual, double range) =>
        new(field, Text(requested, "G4"), Text(actual, "G4"), Math.Min(1, Math.Abs(actual - requested) / range));

    // Категориальный пункт: совпало или нет
    private static SpecDeviation Exact(string field, object? requested, object? actual) =>
        new(field, requested?.ToString() ?? "нет", actual?.ToString() ?? "нет", Equals(requested, actual) ? 0 : 1);

    private static string Yes(bool value) => value ? "да" : "нет";

    private static string Text(double value, string format) => value.ToString(format, CultureInfo.InvariantCulture);
}
