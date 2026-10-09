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
    /// TopK элементов, лидер по метрике первым
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
    /// Ход отдан не лидеру по метрике: ради разведки или потому, что лидер отказал и работу взял
    /// запасной. Без этой пометки нельзя отличить осознанный выбор роутера от жребия при разборе
    /// накопленных ходов.
    /// </summary>
    public bool IsExploration { get; set; }

    /// <summary>
    /// Прогноз качества победителя в момент выбора. По нему калибруется планка: прогноз при нынешних
    /// весах уже видел отзыв на этот ход и обещал бы больше, чем знает.
    /// </summary>
    public double? Forecast { get; set; }

    /// <summary>
    /// Кандидаты, отказавшие на этом ходе (ошибка, таймаут, пустой ответ), по порядку попыток
    /// </summary>
    public List<string> Failed { get; set; } = [];

    /// <summary>
    /// Объем хода не вошел ни в одного кандидата: ход отдан тому, у кого предел больше
    /// </summary>
    public bool ContextShortfall { get; set; }

    /// <summary>
    /// Баллы за задачу. Проставляет судья после того, как победитель ответил.
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Выбор шел с планкой достаточности: истина, если кто-то до нее дотянул, ложь, если ход отдан
    /// сильнейшему при недоборе; пусто, если планки не было.
    /// </summary>
    public bool? BarReached { get; set; }
}
