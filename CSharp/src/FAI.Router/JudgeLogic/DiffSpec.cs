namespace FAI.Router.JudgeLogic;

/// <summary>
/// Расхождение по одному пункту ТЗ
/// </summary>
/// <param name="Field">Название пункта</param>
/// <param name="Requested">Заказано</param>
/// <param name="Actual">Получено</param>
/// <param name="Deviation">Отклонение: 0 — совпало, 1 — полное расхождение</param>
public record SpecDeviation(string Field, string Requested, string Actual, double Deviation);

/// <summary>
/// Разбор расхождений между ТЗ и фактом по каждому пункту (режим критика)
/// </summary>
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
    /// Среднее отклонение по всем пунктам
    /// </summary>
    public double TotalDeviation => Deviations.Average(deviation => deviation.Deviation);

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
    public static DiffSpec Compare(Specifications requested, Specifications actual) =>
        new([
            Exact("Стиль", requested.StyleType, actual.StyleType),
            Number("Объём в символах", requested.SymbolLength, actual.SymbolLength),
            Number("Объём в словах", requested.WordLength, actual.WordLength),
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
            Exact("Ссылки на источники", requested.HasReferences, actual.HasReferences)
        ]);

    /// <summary>
    /// Отчёт критика: список проваленных пунктов
    /// </summary>
    public override string ToString() =>
        string.Join(Environment.NewLine,
            Mismatches.Select(item => $"{item.Field}: заказано {item.Requested}, получено {item.Actual}"));

    // Числовой пункт: относительное отклонение от заказанного
    private static SpecDeviation Number(string field, double requested, double actual)
    {
        // Заказан ноль, получено больше нуля — расхождение полное: делить не на что
        double deviation = requested == 0
            ? (actual == 0 ? 0 : 1)
            : Math.Min(1, Math.Abs(actual - requested) / Math.Abs(requested));

        return new SpecDeviation(field, requested.ToString("G4"), actual.ToString("G4"), deviation);
    }

    // Категориальный пункт: совпало или нет
    private static SpecDeviation Exact(string field, object? requested, object? actual) =>
        new(field, requested?.ToString() ?? "—", actual?.ToString() ?? "—", Equals(requested, actual) ? 0 : 1);
}
