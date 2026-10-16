using FAI.Router.Enums;

namespace FAI.Router.RotationTracking;

/// <summary>
/// Отзыв на результат
/// </summary>
public class Feedback
{
    /// <summary>
    /// Во сколько раз выдуманные факты учат сильнее прочих причин по содержанию. Ошибку человек
    /// замечает и правит сам, выдумку принимает за правду и несет дальше, поэтому она стоит дороже.
    /// Величина не измерялась.
    /// </summary>
    public const double MadeUpWeight = 1.5;

    /// <summary>
    /// Тип фидбека (человеческий стоит дороже и авто-оценщик учится по человеческому)
    /// </summary>
    public FeedbackType FType { get; set; } = FeedbackType.Auto;

    /// <summary>
    /// Агрегированная оценка
    /// </summary>
    public double FeadbackScore { get; set; }

    /// <summary>
    /// Названные причины недовольства. Учитываются только у отрицательного отзыва: силу его шага
    /// задает <see cref="QualityWeight"/>.
    /// </summary>
    public FeedbackReason Reasons { get; set; }

    /// <summary>
    /// Множитель силы отрицательного отзыва по причинам: насколько он говорит о качестве модели на
    /// такой задаче. Без причин единица, как прежде. Только причины формы дают ноль: вектор качества
    /// не двигается, а предпочтение учитывает вызывающий. Причины по содержанию дают единицу,
    /// выдуманные факты <see cref="MadeUpWeight"/>; форма рядом с ними силу не меняет.
    /// </summary>
    /// <param name="reasons">Названные причины</param>
    public static double QualityWeight(FeedbackReason reasons) =>
        (reasons & FeedbackReason.Content) switch
        {
            FeedbackReason.None => reasons == FeedbackReason.None ? 1 : 0,
            var content when content.HasFlag(FeedbackReason.MadeUp) => MadeUpWeight,
            _ => 1
        };
}
