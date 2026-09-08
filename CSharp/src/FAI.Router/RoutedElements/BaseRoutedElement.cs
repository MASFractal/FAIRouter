using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;
using FAI.Router.Enums;
using FAI.Router.RotationTracking;

namespace FAI.Router.RoutedElements;

/// <summary>
/// Базовый элемент для роутинга
/// Соответствует модели
/// </summary>
public class BaseRoutedElement
{
    /// <summary>
    /// Имя элемента
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Вектор для сравнения (обучаемый), инициализация по Ксавье
    /// </summary>
    public Vector IdealMatchVector { get; set; } = XavierVector(Settings.FeaturesDim + Settings.FeaturesSpecDim);

    /// <summary>
    /// Что кандидат умеет. По умолчанию считается, что все: ограничения задает вызывающий.
    /// </summary>
    public Capability Capabilities { get; set; } = Capability.All;

    /// <summary>
    /// Наибольший объем ответа в символах, который кандидат осилит. Ноль означает без ограничения.
    /// </summary>
    public int ContextLimit { get; set; }

    /// <summary>
    /// Число оцененных ходов, доставшихся кандидату. В формуле температуры это m_k.
    /// Заполняется из журнала через SqliteTraceStore.LoadStatistics.
    /// </summary>
    public int Experience { get; set; }

    /// <summary>
    /// Оценка дисперсии отзывов о кандидате. В формуле температуры это D*_k.
    /// Осмысленна начиная с двух ходов, до этого считается неизвестной.
    /// </summary>
    public double ScoreVariance { get; set; }

    /// <summary>
    /// Число токенов в секунду
    /// </summary>
    public double TPS { get; set; } = 1;

    /// <summary>
    /// Цена долларов за 1 млн токенов (вход)
    /// </summary>
    public double DPMTInp { get; set; }

    /// <summary>
    /// Цена долларов за 1 млн токенов (выход)
    /// </summary>
    public double DPMTOutp { get; set; }

    /// <summary>
    /// Получение оценки качества для данного элемента роутинга
    /// по умолчанию скалярное произведение
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public virtual double GetQualityScore(Vector features) =>
        Settings.Center(features).Dot(IdealMatchVector);

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
    public bool Supports(Specifications? specifications, Capability required = Capability.None)
    {
        if (specifications is not null)
        {
            if (specifications.CodeBlockCount > 0)
                required |= Capability.Code;

            if (specifications.FormulaCount > 0)
                required |= Capability.Formulas;

            if (ContextLimit > 0 && specifications.SymbolLength > ContextLimit)
                return false;
        }

        return (Capabilities & required) == required;
    }

    /// <summary>
    /// Стоимость запроса в долларах по ценам кандидата
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    public double GetCost(InputFeatures features) =>
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
    public static Vector XavierVector(int dimension)
    {
        double limit = Math.Sqrt(6.0 / dimension);
        Vector vector = new(dimension);

        for (int i = 0; i < dimension; i++)
            vector[i] = (Random.Shared.NextDouble() * 2 - 1) * limit;

        return vector;
    }
}
