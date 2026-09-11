using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.Persistence;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using FAI.Router.Services;
using FAI.Router.Training;

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
/// <param name="Critic">Разбор расхождений по пунктам</param>
/// <param name="RoundId">Номер хода в журнале, под ним ставится отзыв</param>
public sealed record RouterAnswer(
    string Text,
    BaseRoutedElement Winner,
    Tracert Trace,
    Specifications? Requested,
    Specifications? Actual,
    double? Score,
    DiffSpec? Critic,
    long? RoundId);

/// <summary>
/// Фасад: один объект на весь контур. Выбор исполнителя, выполнение с запасным вариантом,
/// замер ответа, оценка судьи, журнал, отзывы и обучение. Клиент модели для распознавания
/// задания и разбора ответа задается в Settings.LLM.
/// </summary>
public class FaiRouter
{
    private readonly Func<BaseRoutedElement, string, Task<string>> _execute;
    private readonly int _topk;
    private readonly bool _measure;
    private readonly SpecOutputService _measurer = new();
    private readonly RouterTrainer _routerTrainer = new(learningRate: 0.05f);
    private readonly JudgeTrainer _judgeTrainer;

    /// <summary>
    /// Кандидаты на исполнение
    /// </summary>
    public IReadOnlyList<BaseRoutedElement> Candidates { get; }

    /// <summary>
    /// Судья с обучаемой матрицей
    /// </summary>
    public Judge Judge { get; } = new();

    /// <summary>
    /// Журнал ходов; пусто, если база не задана
    /// </summary>
    public SqliteTraceStore? Traces { get; }

    /// <summary>
    /// Хранилище весов; пусто, если база не задана
    /// </summary>
    public SqliteWeightsStore? Weights { get; }

    /// <summary>
    /// Роутер целиком
    /// </summary>
    /// <param name="candidates">Кандидаты на исполнение</param>
    /// <param name="execute">Как получить ответ выбранного кандидата на запрос</param>
    /// <param name="databasePath">Файл весов и журнала; пусто, если память не нужна</param>
    /// <param name="topk">Сколько лучших участвуют в выборе</param>
    /// <param name="measure">Замерять ли ответ судьей; стоит одного обращения к модели на ход</param>
    public FaiRouter(
        IEnumerable<BaseRoutedElement> candidates,
        Func<BaseRoutedElement, string, Task<string>> execute,
        string? databasePath = null,
        int topk = 5,
        bool measure = true)
    {
        Candidates = [.. candidates];
        _execute = execute;
        _topk = topk;
        _measure = measure;
        _judgeTrainer = new JudgeTrainer(Judge, learningRate: 0.5f);

        if (databasePath is null)
            return;

        Traces = new SqliteTraceStore(databasePath);
        Weights = new SqliteWeightsStore(databasePath);
        Load();
    }

    /// <summary>
    /// Один ход по тексту запроса
    /// </summary>
    /// <param name="prompt">Текст запроса</param>
    /// <param name="required">Требования к возможностям, которых нет в задании</param>
    public async Task<RouterAnswer> AskAsync(string prompt, Capability required = Capability.None)
    {
        Tracert trace = await Env.RouteAsync(prompt, Candidates, _topk, required).ConfigureAwait(false);
        string text = await Env.ExecuteAsync(trace, candidate => _execute(candidate, prompt)).ConfigureAwait(false);

        Specifications? actual = null;
        double? score = null;
        DiffSpec? critic = null;

        if (_measure && !string.IsNullOrWhiteSpace(text) && trace.RequestedSpec is not null)
        {
            actual = await _measurer.GetSpecificationsAsync(text).ConfigureAwait(false);
            score = Judge.Rate(trace, trace.RequestedSpec, actual);
            critic = Judge.Criticize(trace.RequestedSpec, actual);
        }

        long? roundId = null;

        if (Traces is not null)
        {
            roundId = Traces.Append(trace, trace.RequestedSpec, actual, prompt);

            // Отзыв критика ставится сразу; человеческий, если придет, его перезапишет
            if (critic is not null)
                Traces.SetFeedback(roundId.Value, new Feedback
                {
                    FType = FeedbackType.Auto,
                    FeadbackScore = 1 - critic.TotalDeviation
                });
        }

        return new RouterAnswer(text, trace.Winner, trace, trace.RequestedSpec, actual, score, critic, roundId);
    }

    /// <summary>
    /// Отзыв на ход: единица означает «нравится», ноль означает «нет»
    /// </summary>
    /// <param name="roundId">Номер хода из RouterAnswer</param>
    /// <param name="score">Оценка от нуля до единицы</param>
    /// <param name="human">Человеческий отзыв или автоматический</param>
    public void Feedback(long roundId, double score, bool human = true) =>
        RequireJournal().SetFeedback(roundId, new Feedback
        {
            FType = human ? FeedbackType.Human : FeedbackType.Auto,
            FeadbackScore = score
        });

    /// <summary>
    /// Обучение по накопленному журналу. Возвращает ошибку последней эпохи.
    /// </summary>
    /// <param name="epochs">Сколько раз пройти по выборке</param>
    public double Train(int epochs = 1)
    {
        SqliteTraceStore journal = RequireJournal();

        // Среднее по задачам задается один раз: веса обучены в пространстве с этим средним
        Settings.TaskMean ??= journal.GetFeatureMean();

        IReadOnlyList<TrainingRound> sample = journal.ReadRated(Candidates);
        double loss = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            loss = 0;

            foreach (TrainingRound round in sample)
            {
                loss += _routerTrainer.Train(round.Trace, round.Feedback);

                // Судья учится только у человека. Автоотзыв это разбор расхождений по пунктам,
                // и учить по нему судью значило бы подгонять одну автоматическую оценку под
                // другую, а человек из этого круга выпадал бы совсем
                if (round.Feedback.FType == FeedbackType.Human && round.Requested is not null && round.Actual is not null)
                    loss += _judgeTrainer.Train(round.Requested, round.Actual, round.Feedback.FeadbackScore);
            }
        }

        journal.LoadStatistics(Candidates);

        return loss;
    }

    /// <summary>
    /// Сохраняет веса, матрицу судьи и среднее по задачам
    /// </summary>
    public void Save()
    {
        SqliteWeightsStore store = RequireWeights();
        store.Save(Candidates);
        store.Save(Judge);

        if (Settings.TaskMean is not null)
            store.SaveTaskMean(Settings.TaskMean);
    }

    /// <summary>
    /// Восстанавливает веса, матрицу судьи, среднее по задачам и опыт кандидатов
    /// </summary>
    public void Load()
    {
        SqliteWeightsStore store = RequireWeights();
        store.Load(Candidates);
        store.Load(Judge);
        Settings.TaskMean = store.LoadTaskMean() ?? Settings.TaskMean;
        Traces?.LoadStatistics(Candidates);
    }

    private SqliteTraceStore RequireJournal() =>
        Traces ?? throw new InvalidOperationException("Журнал не задан: создайте роутер с databasePath.");

    private SqliteWeightsStore RequireWeights() =>
        Weights ?? throw new InvalidOperationException("Хранилище не задано: создайте роутер с databasePath.");
}
