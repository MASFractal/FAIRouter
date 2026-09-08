using AI.DataStructs.Algebraic;
using System.Text.Json.Serialization;
using FAI.Router.Enums;

namespace FAI.Router.JudgeLogic;

/// <summary>
/// Спецификация (ТЗ)
/// </summary>
public class Specifications
{
    // Типичный масштаб поля. Вектор сравнивается косинусом, поэтому координаты обязаны быть
    // соизмеримы: в сырых единицах объем в символах давал 98% нормы вектора, и косинус мерил
    // только длину, из-за чего текст в противоположном стиле получал оценку 0,9998.
    private const double SymbolLengthScale = 20000;
    private const double WordLengthScale = 3000;
    private const double CountScale = 20;
    private const double HeadingDepthScale = 6;
    private const double SentenceLengthScale = 40;
    private const double ReadabilityMax = 100;

    // Доли приходят от модели, а она границы схемы соблюдает не всегда: DeepSeek возвращал
    // termDensity 4 и 80 при объявленных 0-1. Такое значение забивает норму вектора целиком,
    // поэтому границу держит сам тип, а не только схема ответа.
    private double _readabilityScore;
    private double _termDensity;
    private double _formalityScore;

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
    /// Читаемость текста (индекс Флеша-Кинкейда или аналог), 0-100
    /// </summary>
    public double ReadabilityScore
    {
        get => _readabilityScore;
        set => _readabilityScore = Math.Clamp(value, 0, ReadabilityMax);
    }

    /// <summary>
    /// Доля терминологии/иностранных слов, 0-1
    /// </summary>
    public double TermDensity
    {
        get => _termDensity;
        set => _termDensity = Math.Clamp(value, 0, 1);
    }

    /// <summary>
    /// Формальность тона, 0-1
    /// </summary>
    public double FormalityScore
    {
        get => _formalityScore;
        set => _formalityScore = Math.Clamp(value, 0, 1);
    }

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
    /// <remarks>
    /// Вычисляется из остальных полей, поэтому в сохраненный вид не попадает: иначе накопитель
    /// хранил бы одни и те же данные дважды, а при смене масштабов старые копии разошлись бы
    /// с тем, что дает текущий код.
    /// </remarks>
    [JsonIgnore]
    public Vector FeaturesSpecificationVector => GetVector();

    // Формирования вектора признаков
    private Vector GetVector()
    {
        // Language и HasReferences в вектор не включены: строковый и булевый признаки
        // требуют отдельного способа кодирования (сравнение языка, one-hot и т.п.),
        // который пока не определен.
        Vector featuresVector =
        [
            .. Style2Vector(),
            Scaled(SymbolLength, SymbolLengthScale),
            Scaled(WordLength, WordLengthScale),
            Scaled(ParagraphCount, CountScale),
            Scaled(SectionCount, CountScale),
            Scaled(ListItemCount, CountScale),
            Scaled(TableCount, CountScale),
            Scaled(CodeBlockCount, CountScale),
            Scaled(FormulaCount, CountScale),
            Scaled(HeadingDepth, HeadingDepthScale),
            Scaled(AvgSentenceLength, SentenceLengthScale),
            ReadabilityScore / ReadabilityMax,
            TermDensity,
            FormalityScore
        ];
        return featuresVector;
    }

    // Логарифмическая шкала: у объемов и счетчиков значимо отношение величин, а не разница,
    // а деление на масштаб приводит поле к единичному порядку. Отрицательное значение может
    // прийти от модели, распознающей ТЗ, а логарифм на нем дает NaN и портит весь вектор.
    internal static double Scaled(double value, double scale) =>
        Math.Log(1 + Math.Max(0, value)) / Math.Log(1 + scale);

    private Vector Style2Vector()
    {
        int position = (int)StyleType;
        Vector vector = new Vector(Enum.GetValues<Style>().Length);
        vector[position] = 1;
        return vector;
    }
}
