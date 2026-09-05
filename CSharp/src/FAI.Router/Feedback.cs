namespace FAI.Router;

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

/// <summary>
/// Тип фидбека
/// </summary>
public enum FeedbackType 
{
    Human = 1,
    Auto = 2
}
