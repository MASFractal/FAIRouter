using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.Persistence;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using FAI.Router.Training;

namespace FAI.Router;

/// <summary>
/// Память роутера: журнал ходов, отзывы, обучение по ним, веса и калибровка планки.
/// </summary>
/// <remarks>
/// Обучение и сохранение идут под одной блокировкой, а выбор их не ждет: обучение подменяет вектор
/// кандидата целиком, и соседний ход видит либо старый вектор, либо новый.
/// </remarks>
internal sealed class RouterMemory
{
    /// <summary>Сколько последних человеческих отзывов берет калибровка планки</summary>
    private const int CalibrationLimit = 1000;

    private readonly IReadOnlyList<BaseRoutedElement> _candidates;
    private readonly Judge _judge;
    private readonly RouterTrainer _routerTrainer = new(learningRate: 0.05f);
    private readonly JudgeTrainer _judgeTrainer;
    private readonly Lock _gate = new();

    // Кандидаты, чьи векторы обучены или загружены: сохраняются только они
    private readonly HashSet<string> _trained = [];

    // Ходы, обученные после последнего Save: отмечаются в журнале вместе с сохранением весов
    private readonly HashSet<long> _pending = [];

    private CalibrationState? _calibration;

    public RouterMemory(string databasePath, IReadOnlyList<BaseRoutedElement> candidates, Judge judge)
    {
        _candidates = candidates;
        _judge = judge;
        _judgeTrainer = new JudgeTrainer(judge, learningRate: 0.5f);
        Traces = new SqliteTraceStore(databasePath);
        Weights = new SqliteWeightsStore(databasePath);
    }

    public SqliteTraceStore Traces { get; }

    public SqliteWeightsStore Weights { get; }

    /// <summary>Записывает ход и автоотзыв к нему; возвращает номер хода</summary>
    public long Append(Tracert trace, Specifications? actual, string prompt, double? assessment)
    {
        long roundId = Traces.Append(trace, trace.RequestedSpec, actual, prompt);

        // Автоотзыв это итоговая оценка по содержанию и форме; человеческий, если придет, его перезапишет.
        // Нечисловая или вне 0-1 оценка в обучение не идет: сбой судьи это отсутствие отзыва
        if (assessment is >= 0 and <= 1)
            Traces.SetFeedback(roundId, new Feedback { FType = FeedbackType.Auto, FeadbackScore = assessment.Value });

        return roundId;
    }

    /// <summary>Отзыв к ходу; ход возвращается в очередь обучения, калибровка пересчитается</summary>
    public void Feedback(long roundId, double score, bool human)
    {
        Traces.SetFeedback(roundId, new Feedback { FType = human ? FeedbackType.Human : FeedbackType.Auto, FeadbackScore = score });

        lock (_gate)
            _pending.Remove(roundId);

        if (human)
            _calibration = null;
    }

    /// <summary>
    /// Калибровка по журналу, запомненная до следующего человеческого отзыва, обучения или загрузки:
    /// прежде каждый ход читал тысячу строк журнала и заново решал задачу Ньютона
    /// </summary>
    public (Calibration Fit, double Rate, int Count) CurrentCalibration()
    {
        CalibrationState state = _calibration ??= Calibrate(CalibrationPairs());

        return (state.Fit, state.Rate, state.Count);
    }

    public IReadOnlyList<(double Quality, double Score)> CalibrationPairs() => Traces.ReadCalibration(_candidates, CalibrationLimit);

    /// <summary>Учит роутер и судью на ходах, которых еще не учили, в порядке записи</summary>
    public TrainingLoss Train(int epochs)
    {
        lock (_gate)
        {
            TrainingRound[] sample = [.. Traces.ReadRated(_candidates, untrainedOnly: true).Where(round => !_pending.Contains(round.Id))];
            double routerLoss = 0, judgeLoss = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                (routerLoss, judgeLoss) = (0, 0);

                foreach (TrainingRound round in sample)
                {
                    routerLoss += _routerTrainer.Train(round.Trace, round.Feedback);
                    _trained.Add(round.Trace.Winner.Name!);

                    // Судья учится только у человека. Автоотзыв это разбор расхождений по пунктам,
                    // и учить по нему судью значило бы подгонять одну автоматическую оценку под
                    // другую, а человек из этого круга выпадал бы совсем
                    if (round.Feedback.FType == FeedbackType.Human && round.Requested is not null && round.Actual is not null)
                        judgeLoss += _judgeTrainer.Train(round.Requested, round.Actual, round.Feedback.FeadbackScore);
                }
            }

            _pending.UnionWith(sample.Select(round => round.Id));
            Traces.LoadStatistics(_candidates);
            _calibration = null;

            return new TrainingLoss(routerLoss, judgeLoss, sample.Length);
        }
    }

    /// <summary>Сохраняет обученные векторы и судью одной транзакцией, затем отмечает обученные ходы</summary>
    public void Save()
    {
        lock (_gate)
        {
            Weights.Save(_candidates.Where(candidate => candidate.Name is not null && _trained.Contains(candidate.Name)), _judge);
            Traces.MarkTrained(_pending);
            _pending.Clear();
        }
    }

    /// <summary>Загружает векторы со средним задач, судью и опыт кандидатов</summary>
    public void Load()
    {
        lock (_gate)
        {
            _trained.UnionWith(Weights.LoadVectors(_candidates));
            Weights.Load(_judge);
            Traces.LoadStatistics(_candidates);
            _calibration = null;
        }
    }

    private static CalibrationState Calibrate(IReadOnlyList<(double Quality, double Score)> pairs) =>
        pairs.Count == 0
            ? new CalibrationState(default, 0, 0)
            : new CalibrationState(Calibration.Fit(pairs), pairs.Average(pair => pair.Score), pairs.Count);

    private sealed record CalibrationState(Calibration Fit, double Rate, int Count);
}
