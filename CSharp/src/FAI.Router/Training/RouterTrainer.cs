using AI.DataStructs.Algebraic;
using FAI.Router.Enums;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;

namespace FAI.Router.Training;

/// <summary>
/// Контрастивное обучение роутера: вектор соответствия победителя сдвигается так, чтобы на
/// похожих задачах впереди оказывался тот, чей ответ понравился. Учится не абсолютная оценка, а
/// порядок, то есть какой кандидат уместнее именно здесь.
/// </summary>
/// <remarks>
/// <para>
/// Градиент идет только в победителя. Соперников на этом ходе не пробовали, и отзыв о них ничего не
/// говорит: прежде их векторы сдвигались по чужому отзыву, и модель, которую ни разу не звали,
/// проседала от чужих лайков. Их оценки в штрафе неподвижны.
/// </para>
/// <para>
/// Штраф усредняется по соперникам, а не суммируется: иначе ход с пятью соперниками двигал
/// победителя впятеро сильнее хода с одним.
/// </para>
/// <para>
/// Счет идет в double и аналитически: при градиенте в одного победителя производная штрафа это
/// вектор задачи со знаком, и тензоры float32 автоматического дифференцирования здесь не нужны.
/// </para>
/// </remarks>
public class RouterTrainer
{
    /// <summary>
    /// Требуемый отрыв победителя от соперника: без зазора обучение останавливается,
    /// едва порядок стал верным, и разделение остается сколь угодно шатким
    /// </summary>
    private const double Margin = 0.1;

    /// <summary>
    /// Оценка, начиная с которой отзыв считается положительным
    /// </summary>
    private const double LikeThreshold = 0.5;

    private readonly double _learningRate;
    private readonly double _explorationWeight;

    /// <summary>
    /// Контрастивное обучение векторов соответствия
    /// </summary>
    /// <param name="learningRate">Скорость обучения</param>
    /// <param name="explorationWeight">
    /// Множитель шага на ходах, отданных не лидеру (<see cref="Tracert.IsExploration"/>). Единица по
    /// умолчанию: разведочный ход учит так же, как обычный. Больше единицы поднимает голос ходов, где
    /// выбор не совпал с мнением роутера, то есть компенсирует смещение выборки к лидерам.
    /// </param>
    public RouterTrainer(float learningRate = 0.01f, double explorationWeight = 1)
    {
        _learningRate = learningRate;
        _explorationWeight = explorationWeight;
    }

    /// <summary>
    /// Один шаг обучения по трассировке хода. Возвращает ошибку до шага. Вектор победителя
    /// подменяется целиком, а не переписывается по элементам.
    /// </summary>
    /// <param name="trace">Трассировка хода: победитель, соперники, признаки задачи</param>
    /// <param name="feedback">Отзыв на ответ победителя</param>
    public double Train(Tracert trace, Feedback feedback)
    {
        BaseRoutedElement winner = trace.Winner;
        BaseRoutedElement[] rivals = [.. trace.TopKElements.Where(element => element != winner)];

        // Соперников нет: контрастивной паре не из чего взяться
        if (rivals.Length == 0)
            return 0;

        // Тот же вид признаков, что и при подсчете прогноза: иначе вектор учился бы в одном
        // пространстве, а работал в другом
        Vector task = Settings.Center(trace.InputFeatureVector, winner.TaskMean ?? Settings.TaskMean);
        Vector vector = winner.IdealMatchVector;
        double winnerScore = task.Dot(vector);
        bool liked = feedback.FeadbackScore >= LikeThreshold;

        // Сила отзыва, а не только его знак. Оценка ровно на пороге не несет сведений, края несут
        // максимум. Без этого множителя посредственный, но одобренный ответ двигал бы веса так же,
        // как отличный, и разведка теряла бы смысл: соперник отвечает лучше, а обучение не видит
        // разницы между «сойдет» и «отлично».
        double weight = Math.Abs(feedback.FeadbackScore - LikeThreshold) / LikeThreshold;

        if (feedback.FType != FeedbackType.Human)
            weight *= Settings.AutoFeedbackWeight;

        if (trace.IsExploration)
            weight *= _explorationWeight;

        double loss = 0;
        int active = 0;

        foreach (BaseRoutedElement rival in rivals)
        {
            double rivalScore = rival.GetQualityScore(trace.InputFeatureVector);

            // Если понравилось, победитель должен быть выше соперника, если нет, то ниже
            double gap = liked ? winnerScore - rivalScore : rivalScore - winnerScore;
            double penalty = Margin - gap;

            if (!double.IsFinite(penalty) || penalty <= 0)
                continue;

            loss += penalty;
            active++;
        }

        loss *= weight / rivals.Length;

        if (active == 0 || weight <= 0)
            return loss;

        // Производная среднего штрафа по вектору победителя: задача со знаком на долю активных пар
        double step = _learningRate * weight * active / rivals.Length * (liked ? 1 : -1);
        Vector trained = vector + task * step;

        if (trained.All(double.IsFinite))
            winner.IdealMatchVector = trained;

        return loss;
    }
}
