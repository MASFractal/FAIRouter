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
    /// Языки, у которых есть своя серия замеров, кодами ISO 639-1. Язык из списка светит своим
    /// разрядом, любой другой известный язык светит последним, неизвестный не светит ничем.
    /// </summary>
    public static readonly string[] LanguageCodes = ["en", "ru", "zh", "fr", "de", "es", "ja", "ko", "pl"];

    /// <summary>Разрядов под язык: по одному на язык из списка и один на прочие.</summary>
    public static int LanguageDim => LanguageCodes.Length + 1;

    /// <summary>Сколько смысловых пунктов и ограничений берется из заказа: больше удорожает суд и размывает разбор</summary>
    public const int MaxItems = 12;

    /// <summary>
    /// Поля формы, которые запрос может задать явно, именами в схеме ответа модели. Остальные поля
    /// (область, область науки, тип задачи) описывают предмет задачи и сверяются всегда.
    /// </summary>
    public static readonly string[] StatableFields =
    [
        "styleType", "symbolLength", "wordLength", "paragraphCount", "sectionCount", "listItemCount",
        "tableCount", "codeBlockCount", "formulaCount", "headingDepth", "avgSentenceLength",
        "readabilityScore", "termDensity", "formalityScore", "language", "hasReferences", "programmingLanguage"
    ];

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
    /// Читаемость текста, 0-100, чем выше, тем легче: индекс Флеша (Reading Ease) для английского, его адаптация Оборневой для русского
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
    /// Насколько запрос требует экспертной подготовки, 0-1 (серия экспертных запросов)
    /// </summary>
    public double ExpertLevel
    {
        get => _expertLevel;
        set => _expertLevel = Math.Clamp(value, 0, 1);
    }

    /// <summary>
    /// Трудность запроса: доля из семи признаков трудного запроса, 0-1
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

    /// <summary>
    /// Поля формы, которые запрос задал явно, именами из <see cref="StatableFields"/>. Остальные
    /// модель угадала разумным ожиданием, и критик их не сверяет: угаданное требование не
    /// требование. Пусто (null) значит неизвестно: заказ собран кодом или распознан прежней
    /// версией, тогда все поля считаются заданными явно.
    /// </summary>
    public List<string>? ExplicitFields { get; set; }

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
    /// Задано ли поле явно: да, если оно в <see cref="ExplicitFields"/> или список неизвестен
    /// </summary>
    /// <param name="field">Имя поля из <see cref="StatableFields"/></param>
    public bool IsExplicit(string field) =>
        ExplicitFields is null || ExplicitFields.Contains(field, StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Разряд языка в векторе: по списку <see cref="LanguageCodes"/>, последний для прочих,
    /// минус единица для неизвестного
    /// </summary>
    /// <param name="code">Код языка ISO 639-1, можно с регионом (ru-RU) и в любом регистре</param>
    public static int LanguageSlot(string? code)
    {
        if (NormalizeLanguage(code) is not { } normalized)
            return -1;

        int index = Array.IndexOf(LanguageCodes, normalized);

        return index >= 0 ? index : LanguageCodes.Length;
    }

    /// <summary>
    /// Код языка без региона и в нижнем регистре: RU, ru-RU и ru_RU дают ru. Пусто, если языка нет.
    /// </summary>
    /// <param name="code">Код языка от модели или замера</param>
    public static string? NormalizeLanguage(string? code)
    {
        if (string.IsNullOrWhiteSpace(code))
            return null;

        string trimmed = code.Trim();
        int region = trimmed.IndexOfAny(['-', '_']);

        return (region > 0 ? trimmed[..region] : trimmed).ToLowerInvariant();
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
        int index = Convert.ToInt32(value);

        // Значение вне перечисления (число от модели) не роняет вектор и не светит ничем
        if (index >= 0 && index < vector.Count)
            vector[index] = 1;

        return vector;
    }

    // Код предмета задачи: как «один из многих», но первое значение означает «не задано» и
    // разряда не имеет. Задача без предмета получает те же координаты, что и раньше, и оценки
    // судьи на ней не меняются
    private static Vector Subject<TEnum>(TEnum value) where TEnum : struct, Enum
    {
        Vector vector = new(Enum.GetValues<TEnum>().Length - 1);
        int index = Convert.ToInt32(value) - 1;

        if (index >= 0 && index < vector.Count)
            vector[index] = 1;

        return vector;
    }
}
