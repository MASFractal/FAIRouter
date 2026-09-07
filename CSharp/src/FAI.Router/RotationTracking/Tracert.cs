using AI.DataStructs.Algebraic;
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
    /// Баллы за задачу
    /// </summary>
    public double Score { get; set; }
}
