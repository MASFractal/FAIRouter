using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;
using FAI.Router.RoutedElements;

namespace FAI.Router.RotationTracking;

/// <summary>
/// Трассировка хода (для обучения роутера)
/// </summary>
public class Tracert
{
    /// <summary>
    /// Победивший элемент
    /// </summary>
    public required BaseRoutedElement Winner { get; set; }

    /// <summary>
    /// TopK элементов
    /// </summary>
    public required List<BaseRoutedElement> TopKElements { get; set; }

    /// <summary>
    /// Входной вектор признаков (для задачи)
    /// </summary>
    public required Vector InputFeatureVector { get; set; }
    
    /// <summary>
    /// Заказанная спецификация (ТЗ), распознанная при построении признаков.
    /// Судье она нужна для оценки хода: без нее пришлось бы обращаться к модели
    /// повторно за тем же самым.
    /// </summary>
    public Specifications? RequestedSpec { get; set; }

    /// <summary>
    /// Ход отдан не лидеру, а случайному сопернику ради разведки. Без этой пометки нельзя
    /// отличить осознанный выбор роутера от жребия при разборе накопленных ходов.
    /// </summary>
    public bool IsExploration { get; set; }

    /// <summary>
    /// Баллы за задачу. Проставляет судья после того, как победитель ответил.
    /// </summary>
    public double Score { get; set; }
}
