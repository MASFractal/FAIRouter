using FAI.Router.JudgeLogic;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;

namespace FAI.Router;

/// <summary>
/// Итог одного хода
/// </summary>
/// <param name="Text">Ответ исполнителя</param>
/// <param name="Winner">Кто ответил</param>
/// <param name="Trace">Трассировка хода</param>
/// <param name="Requested">Распознанное задание</param>
/// <param name="Actual">Замер ответа; пусто, если замер отключен</param>
/// <param name="Score">Оценка судьи; пусто, если замер отключен</param>
/// <param name="Critic">Разбор расхождений по пунктам формы</param>
/// <param name="RoundId">Номер хода в журнале, под ним ставится отзыв</param>
/// <param name="Content">Оценка содержания; пусто, если замер отключен</param>
/// <param name="Assessment">Итоговая оценка ответа: содержание и форма; пусто, если замер отключен</param>
/// <param name="Reached">Планка достаточности: истина, если кто-то до нее дотянул, ложь, если ход отдан сильнейшему при недоборе и человека стоит предупредить; пусто, если выбор шел без планки</param>
public sealed record RouterAnswer(
    string Text,
    BaseRoutedElement Winner,
    Tracert Trace,
    Specifications? Requested,
    Specifications? Actual,
    double? Score,
    DiffSpec? Critic,
    long? RoundId,
    ContentReview? Content = null,
    double? Assessment = null,
    bool? Reached = null);

/// <summary>
/// Цена модели за миллион токенов в валюте поставщика
/// </summary>
/// <param name="Input">За миллион входных токенов</param>
/// <param name="Output">За миллион выходных токенов</param>
public sealed record Price(double Input, double Output);

/// <summary>
/// Ошибка последней эпохи обучения, раздельно у роутера и у судьи: суммы штрафов по ходам эпохи.
/// Приводится к double суммой обеих, как прежде возвращал <see cref="FaiRouter.Train"/>.
/// </summary>
/// <param name="Router">Сумма контрастивных штрафов роутера</param>
/// <param name="Judge">Сумма квадратов расхождения судьи с человеком</param>
/// <param name="Rounds">Сколько ходов вошло в обучение</param>
public readonly record struct TrainingLoss(double Router, double Judge, int Rounds)
{
    /// <summary>Сумма обеих ошибок: так Train возвращал ее до разделения</summary>
    public static implicit operator double(TrainingLoss loss) => loss.Router + loss.Judge;
}
