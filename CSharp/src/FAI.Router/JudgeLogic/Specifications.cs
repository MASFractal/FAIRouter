using AI.DataStructs.Algebraic;
using FAI.Router.Enums;

namespace FAI.Router.JudgeLogic;

/// <summary>
/// Спецификация (ТЗ)
/// </summary>
public class Specifications
{
    /// <summary>
    /// Распознаваемый тип стиля (стиль - представлен one-hot вектором)
    /// </summary>
    public Style StyleType { get; set; } = Style.Other;

    /// <summary>
    /// Длинна текста в символах
    /// </summary>
    public int SymbolLength { get; set; }

    /// <summary>
    /// Длинна текста в словах
    /// </summary>
    public int WordLength { get; set; }

    #region Структурные метрики

    /// <summary>
    /// Число абзацев
    /// </summary>
    public int ParagraphCount { get; set; }

    /// <summary>
    /// Число разделов (заголовков верхнего уровня)
    /// </summary>
    public int SectionCount { get; set; }

    /// <summary>
    /// Число пунктов списков
    /// </summary>
    public int ListItemCount { get; set; }

    /// <summary>
    /// Число таблиц
    /// </summary>
    public int TableCount { get; set; }

    /// <summary>
    /// Число блоков кода
    /// </summary>
    public int CodeBlockCount { get; set; }

    /// <summary>
    /// Число формул
    /// </summary>
    public int FormulaCount { get; set; }

    /// <summary>
    /// Глубина вложенности заголовков (H1/H2/H3...)
    /// </summary>
    public int HeadingDepth { get; set; }

    #endregion

    #region Лексико-стилевые метрики

    /// <summary>
    /// Средняя длина предложения (в словах)
    /// </summary>
    public double AvgSentenceLength { get; set; }

    /// <summary>
    /// Читаемость текста (индекс Флеша-Кинкейда или аналог)
    /// </summary>
    public double ReadabilityScore { get; set; }

    /// <summary>
    /// Доля терминологии/иностранных слов
    /// </summary>
    public double TermDensity { get; set; }

    /// <summary>
    /// Формальность тона
    /// </summary>
    public double FormalityScore { get; set; }

    #endregion

    #region Соответствие заказу

    /// <summary>
    /// Язык ответа
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Наличие ссылок/источников
    /// </summary>
    public bool HasReferences { get; set; }

    #endregion

    /// <summary>
    /// Вектор признаков
    /// </summary>
    public Vector FeaturesSpecificationVector => GetVector();

    // Формирования вектора признаков
    private Vector GetVector()
    {
        // Language и HasReferences в вектор не включены: строковый и булевый признаки
        // требуют отдельного способа кодирования (сравнение языка, one-hot и т.п.),
        // который пока не определён.
        Vector featuresVector =
        [
            .. Style2Vector(),
            SymbolLength,
            WordLength,
            ParagraphCount,
            SectionCount,
            ListItemCount,
            TableCount,
            CodeBlockCount,
            FormulaCount,
            HeadingDepth,
            AvgSentenceLength,
            ReadabilityScore,
            TermDensity,
            FormalityScore
        ];
        return featuresVector;
    }

    private Vector Style2Vector()
    {
        int position = (int)StyleType;
        Vector vector = new Vector(Enum.GetValues<Style>().Length);
        vector[position] = 1;
        return vector;
    }
}
