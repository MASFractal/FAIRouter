using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;
using FAI.Router.Enums;
using FAI.Router.RotationTracking;

namespace FAI.Router.RoutedElements;

/// <summary>
/// Базовый элемент для роутинга
/// Соответствует модели
/// </summary>
/// <remarks>
/// Числа проверяются при записи: NaN и бесконечность отвергаются сразу, потому что в выборе они
/// отравляли бы стандартизацию всей группы. Отрицательная цена допустима и означает, что поставщик ее
/// не зафиксировал (<see cref="HasKnownPrice"/>): такой кандидат в выборе получает худшую цену группы.
/// </remarks>
public class BaseRoutedElement
{
    private Vector _idealMatchVector = XavierVector(Settings.VectorDim);
    private double _tps = 1;
    private double _dpmtInp;
    private double _dpmtOutp;
    private double _costRatio = 1;

    /// <summary>
    /// Имя элемента
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Вектор для сравнения (обучаемый), инициализация по Ксавье. Обучение подменяет его целиком,
    /// а не переписывает по элементам: выбор на соседнем ходе видит старый вектор или новый, но не смесь.
    /// </summary>
    public Vector IdealMatchVector
    {
        get => _idealMatchVector;
        set => _idealMatchVector = value is not null && value.All(double.IsFinite)
            ? value
            : throw new ArgumentException("Вектор кандидата пуст или содержит нечисловые значения.", nameof(value));
    }

    /// <summary>
    /// Среднее по векторам задач, в пространстве которого построен <see cref="IdealMatchVector"/>.
    /// Пусто, значит общее <see cref="Settings.TaskMean"/> (обычно тоже пустое, и вычитания нет).
    /// </summary>
    /// <remarks>
    /// Среднее принадлежит вектору: начальный вектор по рейтингам подгоняется в пространстве с этим
    /// средним, обучение идет в нем же, и сохраняется оно вместе с вектором.
    /// </remarks>
    public Vector? TaskMean { get; set; }

    /// <summary>
    /// Что кандидат умеет. По умолчанию считается, что все: ограничения задает вызывающий.
    /// </summary>
    public Capability Capabilities { get; set; } = Capability.All;

    /// <summary>
    /// Наибольший объем ОТВЕТА в символах, который кандидат осилит. Ноль означает без ограничения.
    /// Сравнивается с заказанным объемом ответа; окно контекста задает <see cref="ContextWindow"/>.
    /// </summary>
    public int ContextLimit { get; set; }

    /// <summary>
    /// Окно контекста в токенах: вход всего диалога вместе с ожидаемым ответом должен в него
    /// поместиться. Ноль означает, что окно неизвестно, и проверки нет.
    /// </summary>
    public int ContextWindow { get; set; }

    /// <summary>
    /// Условный опыт кандидата: человеческие отзывы целиком, автоотзывы с весом
    /// <see cref="Settings.AutoFeedbackWeight"/>. В формуле температуры это m_k.
    /// Заполняется из журнала через SqliteTraceStore.LoadStatistics.
    /// </summary>
    public double Experience { get; set; }

    /// <summary>
    /// Опыт, который стоят начальные веса: у кандидата с весами по рейтингам он равен
    /// <see cref="Settings.PriorExperience"/>, у случайного или безрейтингового ноль. Входит в
    /// знаменатель температуры, но не в оценку дисперсии.
    /// </summary>
    public double PriorExperience { get; set; }

    /// <summary>
    /// Оценка дисперсии отзывов о кандидате. В формуле температуры это D*_k.
    /// Осмысленна начиная с двух ходов, до этого считается неизвестной.
    /// </summary>
    public double ScoreVariance { get; set; }

    /// <summary>
    /// Число токенов в секунду; ноль и меньше означает, что скорость неизвестна
    /// </summary>
    public double TPS { get => _tps; set => _tps = Finite(value, nameof(TPS)); }

    /// <summary>
    /// Цена за 1 млн токенов (вход); отрицательная означает, что цена неизвестна
    /// </summary>
    public double DPMTInp { get => _dpmtInp; set => _dpmtInp = Finite(value, nameof(DPMTInp)); }

    /// <summary>
    /// Цена за 1 млн токенов (выход); отрицательная означает, что цена неизвестна
    /// </summary>
    public double DPMTOutp { get => _dpmtOutp; set => _dpmtOutp = Finite(value, nameof(DPMTOutp)); }

    /// <summary>
    /// Во сколько раз ход на кандидате обходится дороже его прайса.
    /// </summary>
    /// <remarks>
    /// Прайс считает только объем задачи, а на деле ход дорожает от переделок после отказа
    /// приемки: дешевая модель, которую трижды отправили переделывать, дешева только на бумаге.
    /// Поправку задает вызывающий по накопленным ходам; единица означает «как по прайсу».
    /// Ноль и отрицательные значения бессмысленны и отвергаются.
    /// </remarks>
    public double CostRatio
    {
        get => _costRatio;
        set => _costRatio = Finite(value, nameof(CostRatio)) > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Поправка цены должна быть больше нуля.");
    }

    /// <summary>Цена известна: обе ставки не отрицательны</summary>
    public bool HasKnownPrice => DPMTInp >= 0 && DPMTOutp >= 0;

    /// <summary>
    /// Получение оценки качества для данного элемента роутинга
    /// по умолчанию скалярное произведение
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public virtual double GetQualityScore(Vector features) =>
        Settings.Center(features, TaskMean ?? Settings.TaskMean).Dot(IdealMatchVector);

    /// <summary>
    /// Справится ли кандидат с таким заданием в принципе
    /// </summary>
    /// <remarks>
    /// Проверка идет до сравнения оценок и намеренно грубая: она отсеивает заведомо непригодных,
    /// а не выбирает лучшего. Изображения и инструменты в спецификации не отражены, поэтому такие
    /// требования задает вызывающий отдельным доводом.
    /// </remarks>
    /// <param name="specifications">Заказанная спецификация ответа</param>
    /// <param name="required">Дополнительные требования, которых нет в спецификации</param>
    public bool Supports(Specifications? specifications, Capability required = Capability.None) =>
        Can(specifications, required) && (specifications is null || ContextLimit <= 0 || specifications.SymbolLength <= ContextLimit);

    /// <summary>
    /// Справится ли кандидат с заданием по признакам хода: возможности, объем ответа и окно контекста
    /// для всего диалога вместе с ответом.
    /// </summary>
    /// <param name="features">Признаки хода</param>
    /// <param name="required">Дополнительные требования, которых нет в спецификации</param>
    public bool Supports(InputFeatures features, Capability required = Capability.None) =>
        Supports(features.InputSpecifications, required)
        && (ContextWindow <= 0 || features.InputLen + features.LenAnswer <= ContextWindow);

    /// <summary>
    /// Есть ли у кандидата нужные возможности, без проверки объема
    /// </summary>
    /// <param name="specifications">Заказанная спецификация ответа</param>
    /// <param name="required">Дополнительные требования, которых нет в спецификации</param>
    public bool Can(Specifications? specifications, Capability required = Capability.None)
    {
        if (specifications?.CodeBlockCount > 0)
            required |= Capability.Code;

        if (specifications?.FormulaCount > 0)
            required |= Capability.Formulas;

        return (Capabilities & required) == required;
    }

    /// <summary>
    /// Ожидаемая стоимость запроса: прайс с поправкой на то, во что ход обходится на деле
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public double GetCost(InputFeatures features) => GetListCost(features) * CostRatio;

    /// <summary>
    /// Стоимость запроса по прайсу кандидата, без поправки. С ней сравнивается фактическая
    /// цена хода, поэтому поправка сюда не входит: иначе она считалась бы сама из себя.
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public double GetListCost(InputFeatures features) =>
        (DPMTInp * features.InputLen + DPMTOutp * features.LenAnswer) * 1e-6;

    /// <summary>
    /// Ожидаемое время ответа в секундах
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public double GetTime(InputFeatures features) =>
        features.LenAnswer / (TPS + 0.1);

    /// <summary>
    /// Инициализация обучаемого вектора по Ксавье: равномерно из [-limit, limit],
    /// limit = sqrt(6 / n). Разброс задан размерностью, поэтому прогноз качества на старте
    /// не зависит от того, сколько признаков в векторе.
    /// </summary>
    /// <param name="dimension">Размерность вектора признаков</param>
    /// <param name="random">Генератор; пусто, значит общий</param>
    public static Vector XavierVector(int dimension, Random? random = null)
    {
        double limit = Math.Sqrt(6.0 / dimension);
        Vector vector = new(dimension);
        random ??= Random.Shared;

        for (int i = 0; i < dimension; i++)
            vector[i] = (random.NextDouble() * 2 - 1) * limit;

        return vector;
    }

    private static double Finite(double value, string name) =>
        double.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(name, value, "Нужно конечное число.");
}
