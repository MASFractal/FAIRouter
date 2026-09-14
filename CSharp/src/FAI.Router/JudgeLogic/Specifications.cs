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

    /// <summary>
    /// Языки, у которых на арене свой рейтинг, кодами ISO 639-1. Язык из списка светит своим
    /// разрядом, любой другой известный язык светит последним, неизвестный не светит ничем.
    /// </summary>
    public static readonly string[] LanguageCodes = ["en", "ru", "zh", "fr", "de", "es", "ja", "ko", "pl"];

    /// <summary>Разрядов под язык: по одному на язык арены и один на прочие.</summary>
    public static int LanguageDim => LanguageCodes.Length + 1;

    // Доли приходят от модели, а она границы схемы соблюдает не всегда: DeepSeek возвращал
    // termDensity 4 и 80 при объявленных 0-1. Такое значение забивает норму вектора целиком,
    // поэтому границу держит сам тип, а не только схема ответа.
    private double _readabilityScore;
    private double _termDensity;
    private double _formalityScore;
    private double _expertLevel;
    private double _difficulty;
    private double _factualityDemand;

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

    #region Предмет задачи

    /// <summary>
    /// Предметная область
    /// </summary>
    public Domain Domain { get; set; } = Domain.General;

    /// <summary>
    /// Язык программирования, если заказан или написан код
    /// </summary>
    public ProgrammingLanguage ProgrammingLanguage { get; set; } = ProgrammingLanguage.None;

    /// <summary>
    /// Область науки, если задача научная
    /// </summary>
    public ScienceField ScienceField { get; set; } = ScienceField.None;

    /// <summary>
    /// Тип задачи: что заказчик хочет получить на выходе (письмо, отчет, лендинг, код)
    /// </summary>
    public TaskKind TaskKind { get; set; } = TaskKind.None;

    #endregion

    #region Требования заказа вне вектора ответа

    /// <summary>
    /// Насколько запрос требует экспертной подготовки, 0-1 (категория арены Expert)
    /// </summary>
    public double ExpertLevel
    {
        get => _expertLevel;
        set => _expertLevel = Math.Clamp(value, 0, 1);
    }

    /// <summary>
    /// Трудность запроса: доля из семи признаков трудного запроса арены, 0-1 (Hard Prompts)
    /// </summary>
    public double Difficulty
    {
        get => _difficulty;
        set => _difficulty = Math.Clamp(value, 0, 1);
    }

    /// <summary>
    /// Насколько ответ держится на проверяемых фактах, 0-1: чем выше, тем дороже ошибка в факте
    /// </summary>
    public double FactualityDemand
    {
        get => _factualityDemand;
        set => _factualityDemand = Math.Clamp(value, 0, 1);
    }

    /// <summary>
    /// Смысловые пункты, которые ответ обязан раскрыть по сути: что сравнить, посчитать, решить.
    /// По ним судья содержания проверяет полноту, а не число разделов.
    /// </summary>
    public List<string> RequiredPoints { get; set; } = [];

    /// <summary>
    /// Явные ограничения запроса, выполнение которых можно проверить (Instruction Following)
    /// </summary>
    public List<string> Constraints { get; set; } = [];

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

    /// <summary>
    /// Разряд языка в векторе: по списку <see cref="LanguageCodes"/>, последний для прочих,
    /// минус единица для неизвестного
    /// </summary>
    /// <param name="code">Код языка ISO 639-1</param>
    public static int LanguageSlot(string? code)
    {
        if (string.IsNullOrWhiteSpace(code))
            return -1;

        int index = Array.IndexOf(LanguageCodes, code.Trim().ToLowerInvariant());

        return index >= 0 ? index : LanguageCodes.Length;
    }

    // Формирования вектора признаков
    private Vector GetVector()
    {
        // Предмет задачи, язык и ссылки идут кодом «один из многих» и признаком 0/1: по ним
        // роутер учит, кто в какой области силен, а масштабов у них нет, как и у стиля
        Vector featuresVector =
        [
            .. OneHot(StyleType),
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
            FormalityScore,
            .. Subject(Domain),
            .. Subject(ProgrammingLanguage),
            .. Subject(ScienceField),
            .. Subject(TaskKind),
            .. LanguageVector(),
            HasReferences ? 1 : 0
        ];
        return featuresVector;
    }

    // Логарифмическая шкала: у объемов и счетчиков значимо отношение величин, а не разница,
    // а деление на масштаб приводит поле к единичному порядку. Отрицательное значение может
    // прийти от модели, распознающей ТЗ, а логарифм на нем дает NaN и портит весь вектор.
    internal static double Scaled(double value, double scale) =>
        Math.Log(1 + Math.Max(0, value)) / Math.Log(1 + scale);

    private Vector LanguageVector()
    {
        Vector vector = new(LanguageDim);
        int slot = LanguageSlot(Language);

        if (slot >= 0)
            vector[slot] = 1;

        return vector;
    }

    // Код «один из многих» по перечислению: разряд по порядку значения
    private static Vector OneHot<TEnum>(TEnum value) where TEnum : struct, Enum
    {
        Vector vector = new(Enum.GetValues<TEnum>().Length);
        vector[Convert.ToInt32(value)] = 1;
        return vector;
    }

    // Код предмета задачи: как «один из многих», но первое значение означает «не задано» и
    // разряда не имеет. Задача без предмета получает те же координаты, что и раньше, и оценки
    // судьи на ней не меняются
    private static Vector Subject<TEnum>(TEnum value) where TEnum : struct, Enum
    {
        Vector vector = new(Enum.GetValues<TEnum>().Length - 1);
        int index = Convert.ToInt32(value) - 1;

        if (index >= 0)
            vector[index] = 1;

        return vector;
    }
}
