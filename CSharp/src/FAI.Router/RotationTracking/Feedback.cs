using FAI.Router.Enums;

namespace FAI.Router.RotationTracking;

/// <summary>
/// Отзыв на результат
/// </summary>
public class Feedback
{

    /// <summary>
    /// Тип фидбека (человеческий стоит дороже и авто-оценщик учится по человеческому)
    /// </summary>
    public FeedbackType FType { get; set; } = FeedbackType.Auto;

    /// <summary>
    /// Агрегированная оценка
    /// </summary>
    public double FeadbackScore { get; set; }
}
