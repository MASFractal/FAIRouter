using AI.LLM.Core.Models.Common.Messages;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.LLM;
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
/// <param name="Critic">Разбор расхождений по пунктам формы</param>
/// <param name="RoundId">Номер хода в журнале, под ним ставится отзыв</param>
/// <param name="Content">Оценка содержания; пусто, если замер отключен</param>
/// <param name="Assessment">Итоговая оценка ответа: содержание и форма; пусто, если замер отключен</param>
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
    double? Assessment = null);

/// <summary>
/// Фасад: один объект на весь контур. Выбор исполнителя, выполнение с запасным вариантом,
/// замер ответа, оценка судьи, журнал, отзывы и обучение. Клиент модели для распознавания
/// задания и разбора ответа задается в Settings.LLM.
/// </summary>
public class FaiRouter
{
    private readonly Func<BaseRoutedElement, IReadOnlyList<LLMMessage>, Task<string>> _execute;
    private readonly int _topk;
    private readonly bool _measure;
    private readonly SpecOutputService _measurer = new();
    private readonly ContentJudge _contentJudge;
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
    /// <param name="execute">Как получить ответ выбранного кандидата на диалог: он видит все реплики, а не одну</param>
    /// <param name="databasePath">Файл весов и журнала; пусто, если память не нужна</param>
    /// <param name="topk">Сколько лучших участвуют в выборе</param>
    /// <param name="measure">Оценивать ли ответ по форме и содержанию; стоит двух обращений к модели на ход</param>
    /// <param name="contentJudge">Свой судья содержания, например с проверкой фактов по вебу; не задан, тогда общий</param>
    public FaiRouter(
        IEnumerable<BaseRoutedElement> candidates,
        Func<BaseRoutedElement, IReadOnlyList<LLMMessage>, Task<string>> execute,
        string? databasePath = null,
        int topk = 5,
        bool measure = true,
        ContentJudge? contentJudge = null)
    {
        Candidates = [.. candidates];
        _execute = execute;
        _topk = topk;
        _measure = measure;
        _contentJudge = contentJudge ?? new ContentJudge();
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
    public Task<RouterAnswer> AskAsync(string prompt, Capability required = Capability.None) =>
        AskAsync([new LLMMessage(LLMMessage.UserRole, prompt)], required);

    /// <summary>
    /// Один ход по диалогу: задача распознается по последнему сообщению пользователя, а
    /// исполнителю уходит весь диалог целиком. То же, что ask_messages в версии на Python.
    /// </summary>
    /// <param name="messages">Реплики диалога по порядку</param>
    /// <param name="required">Требования к возможностям, которых нет в задании</param>
    public async Task<RouterAnswer> AskAsync(IEnumerable<LLMMessage> messages, Capability required = Capability.None)
    {
        LLMMessage[] dialog = [.. messages];

        // Задание распознается по тексту, поэтому картинка или иное содержимое без текста задачей не считается
        string prompt = dialog
            .LastOrDefault(message => string.Equals(message.Role, LLMMessage.UserRole, StringComparison.OrdinalIgnoreCase))
            ?.Content as string ?? "";

        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("В диалоге нет сообщения пользователя, задачу распознать не из чего.", nameof(messages));

        int turns = dialog.Count(message => string.Equals(message.Role, LLMMessage.UserRole, StringComparison.OrdinalIgnoreCase));
        Tracert trace = await Env.RouteAsync(prompt, Candidates, _topk, required, turns: turns).ConfigureAwait(false);
        string text = await Env.ExecuteAsync(trace, candidate => _execute(candidate, dialog)).ConfigureAwait(false);

        Specifications? actual = null;
        double? score = null;
        DiffSpec? critic = null;
        ContentReview? content = null;
        double? assessment = null;

        if (_measure && !string.IsNullOrWhiteSpace(text) && trace.RequestedSpec is not null)
        {
            // Форма и содержание не зависят друг от друга: замер структуры и суд содержания идут разом
            Task<Specifications> measuring = _measurer.GetSpecificationsAsync(text);
            Task<ContentReview> reviewing = _contentJudge.ReviewAsync(prompt, trace.RequestedSpec, text);

            actual = await measuring.ConfigureAwait(false);
            content = await reviewing.ConfigureAwait(false);
            score = Judge.Rate(trace, trace.RequestedSpec, actual);
            critic = Judge.Criticize(trace.RequestedSpec, actual, content);
            assessment = Judge.Assess(critic, content);
        }

        long? roundId = null;

        if (Traces is not null)
        {
            roundId = Traces.Append(trace, trace.RequestedSpec, actual, prompt);

            // Автоотзыв это итоговая оценка по содержанию и форме; человеческий, если придет, его перезапишет
            if (assessment is not null)
                Traces.SetFeedback(roundId.Value, new Feedback
                {
                    FType = FeedbackType.Auto,
                    FeadbackScore = assessment.Value
                });
        }

        return new RouterAnswer(text, trace.Winner, trace, trace.RequestedSpec, actual, score, critic, roundId, content, assessment);
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
