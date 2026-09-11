using AI.ML.NeuralNetworks.V2;
using AI.ML.NeuralNetworks.V2.Nn;
using AI.ML.NeuralNetworks.V2.Optim;
using AI.ML.NeuralNetworks.V2.Ops;
using FAI.Router.Enums;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;

namespace FAI.Router.Training;

/// <summary>
/// Контрастивное обучение роутера: вектор соответствия победителя и его соперников
/// разводится так, чтобы на похожих задачах впереди оказывался тот, чей ответ понравился.
/// Учится не абсолютная оценка, а порядок, то есть какой кандидат уместнее именно здесь.
/// </summary>
public class RouterTrainer
{
    /// <summary>
    /// Требуемый отрыв победителя от соперника: без зазора обучение останавливается,
    /// едва порядок стал верным, и разделение остается сколь угодно шатким
    /// </summary>
    private const float Margin = 0.1f;

    /// <summary>
    /// Оценка, начиная с которой отзыв считается положительным
    /// </summary>
    private const double LikeThreshold = 0.5;

    /// <summary>
    /// Во сколько раз автоотзыв слабее человеческого.
    /// </summary>
    /// <remarks>
    /// Автоотзыв это разбор расхождений по пунктам, и половина его пунктов расходится почти
    /// всегда: заказ по ним угадан моделью, факт посчитан. Учить по нему наравне с человеком
    /// значило бы давать шуму тот же голос, что и оценке. Совсем не учить тоже нельзя: до
    /// первого человеческого отзыва роутер иначе не учится вовсе. Величина не измерялась.
    /// </remarks>
    private const float AutoFeedbackWeight = 0.25f;

    private readonly float _learningRate;

    /// <summary>
    /// Контрастивное обучение векторов соответствия
    /// </summary>
    /// <param name="learningRate">Скорость обучения</param>
    public RouterTrainer(float learningRate = 0.01f) => _learningRate = learningRate;

    /// <summary>
    /// Один шаг обучения по трассировке хода. Возвращает ошибку до шага.
    /// </summary>
    /// <param name="trace">Трассировка хода: победитель, соперники, признаки задачи</param>
    /// <param name="feedback">Отзыв на ответ победителя</param>
    public double Train(Tracert trace, Feedback feedback)
    {
        BaseRoutedElement[] rivals = [.. trace.TopKElements.Where(element => element != trace.Winner)];

        // Соперников нет: контрастивной паре не из чего взяться
        if (rivals.Length == 0)
            return 0;

        // Тот же вид признаков, что и при подсчете прогноза: иначе вектор учился бы в одном
        // пространстве, а работал в другом
        Tensor task = TensorBridge.ToColumn(Settings.Center(trace.InputFeatureVector));
        bool liked = feedback.FeadbackScore >= LikeThreshold;

        // Параметры строятся на шаг от НЫНЕШНИХ векторов кандидатов, без кэша между ходами.
        // Кэш переставал быть вектором кандидата после первой же загрузки весов или подстановки
        // приора: объект вектора подменялся, а обучение продолжало писать поверх него устаревшие
        // значения. Оптимизатор по той же причине на шаг: состав кандидатов меняется от хода к
        // ходу, а у спуска без импульса нет состояния, которое стоило бы переносить
        (BaseRoutedElement Element, Parameter Vector)[] trained =
            [.. rivals.Prepend(trace.Winner).Select(element => (element, new Parameter(TensorBridge.ToRow(element.IdealMatchVector))))];

        Optimizer optimizer = new SGD([.. trained.Select(item => item.Vector)], lr: _learningRate);
        optimizer.ZeroGrad();

        // Сила отзыва, а не только его знак. Оценка ровно на пороге не несет сведений, края несут
        // максимум. Без этого множителя посредственный, но одобренный ответ двигал бы веса так же,
        // как отличный, и разведка теряла бы смысл: соперник отвечает лучше, а обучение не видит
        // разницы между «сойдет» и «отлично».
        float weight = (float)(Math.Abs(feedback.FeadbackScore - LikeThreshold) / LikeThreshold);

        if (feedback.FType != FeedbackType.Human)
            weight *= AutoFeedbackWeight;

        Tensor loss = TensorOps.MulScalar(GetLoss(trained[0].Vector, trained[1..], task, liked), weight);
        loss.Backward();
        optimizer.Step();

        // Роутер считает по Vector кандидата: без обратной записи обучение на выбор не повлияет
        foreach ((BaseRoutedElement element, Parameter vector) in trained)
            TensorBridge.WriteBack(vector.Tensor, element.IdealMatchVector);

        return TensorBridge.Scalar(loss);
    }

    // Сумма штрафов за недостаточный отрыв победителя от каждого соперника
    private static Tensor GetLoss(
        Parameter winner,
        (BaseRoutedElement Element, Parameter Vector)[] rivals,
        Tensor task,
        bool liked)
    {
        Tensor winnerScore = GetScore(winner, task);
        Tensor? loss = null;

        foreach ((_, Parameter vector) in rivals)
        {
            Tensor rivalScore = GetScore(vector, task);

            // Если понравилось, победитель должен быть выше соперника, если нет, то ниже
            Tensor gap = liked
                ? TensorOps.Sub(winnerScore, rivalScore)
                : TensorOps.Sub(rivalScore, winnerScore);

            Tensor penalty = TensorOps.Relu(TensorOps.AddScalar(TensorOps.Neg(gap), Margin));
            loss = loss is null ? penalty : TensorOps.Add(loss, penalty);
        }

        return loss!;
    }

    // Прогноз качества кандидата: скалярное произведение признаков задачи на его вектор
    private static Tensor GetScore(Parameter vector, Tensor task) =>
        vector.Tensor.MatMul(task).Reshape(1);
}
