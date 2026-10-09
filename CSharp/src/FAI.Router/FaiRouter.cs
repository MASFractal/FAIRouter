using System.Collections.Concurrent;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Services.LLM;
using FAI.Router.Catalog;
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
/// Фасад: один объект на весь контур. Выбор исполнителя, выполнение с запасным вариантом,
/// замер ответа, оценка судьи, журнал, отзывы и обучение. Клиент модели для распознавания
/// задания и разбора ответа принадлежит роутеру (довод llm); не задан, тогда общий Settings.LLM.
/// </summary>
/// <remarks>
/// Память (журнал, обучение, веса, калибровка) живет в <see cref="RouterMemory"/>, фасад ведет ход.
/// </remarks>
public class FaiRouter
{
    private readonly Func<BaseRoutedElement, IReadOnlyList<LLMMessage>, CancellationToken, Task<string>> _execute;
    private readonly int _topk;
    private readonly bool _measure;
    private readonly RouteWeights? _weights;
    private readonly double? _bar;
    private readonly double? _temperatureScale;
    private readonly int _minRatings;
    private readonly SpecInputService? _recognizer;
    private readonly SpecOutputService _measurer;
    private readonly ContentJudge _contentJudge;
    private readonly Random _random;
    private readonly ConcurrentDictionary<string, DateTimeOffset> _failedUntil = new();
    private readonly RouterMemory? _memory;

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
    public SqliteTraceStore? Traces => _memory?.Traces;

    /// <summary>
    /// Хранилище весов; пусто, если база не задана
    /// </summary>
    public SqliteWeightsStore? Weights => _memory?.Weights;

    /// <summary>
    /// Готовая планка со своей калибровкой на все ходы. Задана, тогда уровень из конструктора и
    /// калибровка по журналу не используются.
    /// </summary>
    public SufficiencyBar? Bar { get; set; }

    /// <summary>
    /// На сколько отказавший кандидат (ошибка, таймаут, пустой ответ) выпадает из выбора этого роутера.
    /// Без паузы модель, отвечающая ошибкой сервера, выбиралась снова на каждом ходе.
    /// </summary>
    public TimeSpan FailureCooldown { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Роутер целиком; исполнитель без токена отмены
    /// </summary>
    /// <param name="candidates">Кандидаты на исполнение</param>
    /// <param name="execute">Как получить ответ выбранного кандидата на диалог: он видит все реплики, а не одну</param>
    /// <param name="databasePath">Файл весов и журнала; пусто, если память не нужна</param>
    /// <param name="topk">Сколько лучших участвуют в выборе</param>
    /// <param name="measure">Оценивать ли ответ по форме и содержанию; стоит двух обращений к модели на ход</param>
    /// <param name="contentJudge">Свой судья содержания, например с проверкой фактов по вебу; не задан, тогда общий</param>
    /// <param name="weights">Профиль весов на все ходы: RouteWeights.Quality, Balance, Price или свои; пусто, тогда веса из Settings</param>
    /// <param name="bar">Планка достаточности на все ходы: обязательная вероятность лайка от 0 до 1, калибруется по журналу человеческих отзывов; пусто, тогда выбор по метрике R</param>
    /// <param name="minRatings">Сколько человеческих отзывов нужно, чтобы калибровка планки считалась осмысленной; до этого ход идет без планки</param>
    /// <param name="temperatureScale">Свой множитель температуры роутера для весов без своего множителя; пусто, тогда общий из Settings</param>
    public FaiRouter(
        IEnumerable<BaseRoutedElement> candidates,
        Func<BaseRoutedElement, IReadOnlyList<LLMMessage>, Task<string>> execute,
        string? databasePath = null,
        int topk = 5,
        bool measure = true,
        ContentJudge? contentJudge = null,
        RouteWeights? weights = null,
        double? bar = null,
        int minRatings = 3,
        double? temperatureScale = null)
        : this(candidates, (candidate, messages, _) => execute(candidate, messages), databasePath, topk, measure, contentJudge, weights, bar, minRatings, temperatureScale)
    {
    }

    /// <summary>
    /// Роутер целиком; исполнитель получает токен отмены хода
    /// </summary>
    /// <param name="candidates">Кандидаты на исполнение</param>
    /// <param name="execute">Как получить ответ выбранного кандидата на диалог с токеном отмены хода</param>
    /// <param name="databasePath">Файл весов и журнала; пусто, если память не нужна</param>
    /// <param name="topk">Сколько лучших участвуют в выборе</param>
    /// <param name="measure">Оценивать ли ответ по форме и содержанию; стоит двух обращений к модели на ход</param>
    /// <param name="contentJudge">Свой судья содержания; не задан, тогда общий с клиентом llm</param>
    /// <param name="weights">Профиль весов на все ходы; пусто, тогда веса из Settings</param>
    /// <param name="bar">Планка достаточности на все ходы, вероятность лайка от 0 до 1; пусто, тогда выбор по метрике R</param>
    /// <param name="minRatings">Сколько человеческих отзывов нужно для калибровки планки</param>
    /// <param name="temperatureScale">Свой множитель температуры для весов без своего множителя; пусто, тогда общий</param>
    /// <param name="llm">Клиент модели для распознавания задания и разбора ответа; пусто, тогда общий Settings.LLM</param>
    /// <param name="seed">Зерно жребия выбора; пусто, значит случайное</param>
    public FaiRouter(
        IEnumerable<BaseRoutedElement> candidates,
        Func<BaseRoutedElement, IReadOnlyList<LLMMessage>, CancellationToken, Task<string>> execute,
        string? databasePath = null,
        int topk = 5,
        bool measure = true,
        ContentJudge? contentJudge = null,
        RouteWeights? weights = null,
        double? bar = null,
        int minRatings = 3,
        double? temperatureScale = null,
        LLMBase? llm = null,
        int? seed = null)
    {
        if (bar is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(bar), bar, "Планка вне диапазона: нужна вероятность лайка от 0 до 1.");

        _weights = weights;
        _bar = bar;
        _minRatings = minRatings;
        _temperatureScale = temperatureScale;
        Candidates = [.. candidates];
        _execute = execute;
        _topk = topk;
        _measure = measure;
        _recognizer = llm is null ? null : new SpecInputService(llm);
        _measurer = new SpecOutputService(llm);
        _contentJudge = contentJudge ?? new ContentJudge(llm);
        _random = seed is null ? new Random() : new Random(seed.Value);

        if (_measure && _contentJudge.Model is { } judgeModel && Candidates.Any(candidate => candidate.Name == judgeModel))
            System.Diagnostics.Trace.TraceWarning($"FAIRouter: судья содержания {judgeModel} среди кандидатов и будет судить сам себя; задайте ему другую модель (contentJudge).");

        if (databasePath is not null)
            (_memory = new RouterMemory(databasePath, Candidates, Judge)).Load();
    }

    /// <summary>
    /// Модель-судья по умолчанию: распознает задание и разбирает ответ
    /// </summary>
    public const string DefaultJudgeModel = "openai/gpt-4o-mini";

    /// <summary>
    /// Роутер над моделями FractalRouter (fractalrouter.ru): цены и возможности из его каталога, ответы
    /// кандидатов и работа судьи через него же, одним ключом. То же, что from_fractalrouter в версии на Python.
    /// Доводы те же, что у <see cref="FromOpenAiCompatibleAsync"/>; цены в prices в рублях.
    /// </summary>
    public static Task<FaiRouter> FromFractalRouterAsync(
        string apiKey,
        IEnumerable<string>? modelIds = null,
        string? databasePath = null,
        IReadOnlyDictionary<string, Price>? prices = null,
        string judgeModel = DefaultJudgeModel,
        string baseUrl = Providers.FractalRouter,
        IReadOnlyDictionary<string, double>? tokensPerSecond = null,
        BenchmarkSnapshot? benchmarks = null,
        int topk = 5,
        bool measure = true,
        ContentJudge? contentJudge = null,
        RouteWeights? weights = null,
        double? bar = null,
        CancellationToken cancellationToken = default) =>
        FromOpenAiCompatibleAsync(baseUrl, apiKey, modelIds, databasePath, prices, judgeModel, tokensPerSecond, benchmarks,
            token => ModelCatalog.FetchFractalRouterAsync(apiKey, cancellationToken: token), topk, measure, contentJudge, weights, bar, cancellationToken);

    /// <summary>
    /// Роутер над моделями OpenRouter: цены и возможности из его каталога, ответы через него же.
    /// То же, что from_openrouter в версии на Python. Доводы те же, что у <see cref="FromOpenAiCompatibleAsync"/>;
    /// цены в prices в долларах.
    /// </summary>
    public static Task<FaiRouter> FromOpenRouterAsync(
        string apiKey,
        IEnumerable<string>? modelIds = null,
        string? databasePath = null,
        string judgeModel = DefaultJudgeModel,
        IReadOnlyDictionary<string, double>? tokensPerSecond = null,
        IReadOnlyDictionary<string, Price>? prices = null,
        BenchmarkSnapshot? benchmarks = null,
        int topk = 5,
        bool measure = true,
        ContentJudge? contentJudge = null,
        RouteWeights? weights = null,
        double? bar = null,
        CancellationToken cancellationToken = default) =>
        FromOpenAiCompatibleAsync(Providers.OpenRouter, apiKey, modelIds, databasePath, prices, judgeModel, tokensPerSecond, benchmarks,
            token => ModelCatalog.FetchAsync(cancellationToken: token), topk, measure, contentJudge, weights, bar, cancellationToken);

    /// <summary>
    /// Роутер над любым поставщиком по протоколу OpenAI chat completions: адрес вида https://host/v1,
    /// ключ уходит заголовком Bearer. Через того же поставщика и тем же ключом работает модель-судья.
    /// </summary>
    /// <remarks>
    /// Цены кандидатов берутся из каталога catalogFetch (по умолчанию каталог OpenRouter), а prices их
    /// задает или заменяет. Модель без цены ни там, ни там это ошибка: без цены роутеру нечего
    /// взвешивать. Каталог недоступен по сети, тогда цены берутся из комплекта сборки. Начальные веса
    /// кандидатов берутся из снимка замеров, поэтому роутер небесполезен с первого хода, а обучение на
    /// отзывах его уточняет в том же пространстве признаков: среднее задач кандидат хранит при себе
    /// (<see cref="BaseRoutedElement.TaskMean"/>), и ни загрузка, ни обучение его не меняют. Клиент
    /// судьи принадлежит роутеру; общий Settings.LLM фабрика ставит, только если он еще не задан.
    /// </remarks>
    /// <param name="baseUrl">Адрес поставщика вида https://host/v1</param>
    /// <param name="apiKey">Ключ поставщика</param>
    /// <param name="modelIds">Идентификаторы моделей-кандидатов; один элемент ModelCatalog.Popular или ModelCatalog.All задает набор; пусто, значит популярные</param>
    /// <param name="databasePath">Файл весов и журнала; пусто, если память не нужна</param>
    /// <param name="prices">Цены за миллион токенов для моделей, которых нет в каталоге или чью цену надо заменить</param>
    /// <param name="judgeModel">Модель-судья, работает через тот же ключ</param>
    /// <param name="tokensPerSecond">Скорость по идентификаторам</param>
    /// <param name="benchmarks">Снимок замеров для начальных весов; пусто, значит снимок из комплекта сборки</param>
    /// <param name="catalogFetch">Как получить каталог поставщика; пусто, значит каталог OpenRouter</param>
    /// <param name="topk">Сколько лучших участвуют в выборе</param>
    /// <param name="measure">Оценивать ли ответ по форме и содержанию</param>
    /// <param name="contentJudge">Свой судья содержания; не задан, тогда общий</param>
    /// <param name="weights">Профиль весов на все ходы: RouteWeights.Quality, Balance, Price или свои; пусто, тогда веса из Settings</param>
    /// <param name="bar">Планка достаточности на все ходы: обязательная вероятность лайка от 0 до 1, калибруется по журналу человеческих отзывов</param>
    /// <param name="cancellationToken">Токен отмены создания роутера; ходы отменяются своими токенами</param>
    public static async Task<FaiRouter> FromOpenAiCompatibleAsync(
        string baseUrl,
        string apiKey,
        IEnumerable<string>? modelIds = null,
        string? databasePath = null,
        IReadOnlyDictionary<string, Price>? prices = null,
        string judgeModel = DefaultJudgeModel,
        IReadOnlyDictionary<string, double>? tokensPerSecond = null,
        BenchmarkSnapshot? benchmarks = null,
        Func<CancellationToken, Task<IReadOnlyList<ModelInfo>>>? catalogFetch = null,
        int topk = 5,
        bool measure = true,
        ContentJudge? contentJudge = null,
        RouteWeights? weights = null,
        double? bar = null,
        CancellationToken cancellationToken = default)
    {
        string[] requested = modelIds is null ? [ModelCatalog.Popular] : [.. modelIds.Distinct()];
        prices ??= new Dictionary<string, Price>();
        benchmarks ??= BenchmarkSnapshot.LoadEmbedded();

        // Снимок с чужими именами серий тише пустого: приора не получил бы никто, и роутер молча
        // стартовал бы с уровня поля, уверенный, что рейтинги у него есть
        if (benchmarks.Entries.Count > 0 && BenchmarkPrior.KnownSeries(benchmarks) == 0)
            throw new ArgumentException(
                "В снимке замеров нет ни одной серии из профилей: он собран другой версией библиотеки. "
                + "Пересоберите снимок или возьмите снимок из комплекта (benchmarks: null).", nameof(benchmarks));

        // Снимок старше полугода описывает прошлое поколение моделей
        if (benchmarks.Age?.TotalDays > 180)
            System.Diagnostics.Trace.TraceWarning($"FAIRouter: снимку замеров {benchmarks.Age.Value.Days} дней, начальные веса устарели; пересоберите снимок.");

        Dictionary<string, ModelInfo> known = [];
        bool named = requested.Length == 1 && (string.Equals(requested[0], ModelCatalog.All, StringComparison.OrdinalIgnoreCase)
            || string.Equals(requested[0], ModelCatalog.Popular, StringComparison.OrdinalIgnoreCase));

        if (named || requested.Any(id => !prices.ContainsKey(id)))
        {
            IReadOnlyList<ModelInfo> catalog = await ModelCatalog.FetchOrPopularAsync(
                catalogFetch ?? (token => ModelCatalog.FetchAsync(cancellationToken: token)), cancellationToken).ConfigureAwait(false);

            foreach (ModelInfo model in catalog)
                known.TryAdd(model.Id, model);
        }

        (IReadOnlyList<string> ids, bool _) = ModelCatalog.Select(requested, known);

        if (ids.Count == 0)
            throw new ArgumentException("Список моделей пуст: роутеру не из кого выбирать.", nameof(modelIds));

        List<BaseRoutedElement> candidates = ModelCatalog.CreateCandidates(ids, known, prices, tokensPerSecond, benchmarks, Settings.TaskMean);
        OpenAiCompatibleLlm judge = new(baseUrl, apiKey, judgeModel);

        // Общий клиент остается запасным для компонентов, собранных без клиента; чужой не затирается
        if (!Settings.HasLLM)
            Settings.LLM = judge;

        ConcurrentDictionary<string, LLMBase> clients = new();

        Task<string> Execute(BaseRoutedElement candidate, IReadOnlyList<LLMMessage> messages, CancellationToken token)
        {
            string name = candidate.Name ?? throw new InvalidOperationException("Кандидат без имени: не у кого спрашивать.");
            LLMBase client = clients.GetOrAdd(name, id => new OpenAiCompatibleLlm(baseUrl, apiKey, id));

            return client.SendToLLM(messages, cancellationToken: token);
        }

        // Все кандидаты стартуют с начальных весов по рейтингам: долгая разведка не нужна, и множитель
        // температуры берется пониженный. Хотя бы один без рейтингов, тогда общий
        bool informed = candidates.All(candidate => candidate.PriorExperience > 0);

        return new FaiRouter(candidates, Execute, databasePath, topk, measure, contentJudge, weights, bar,
            temperatureScale: informed ? Settings.PriorTemperatureScale : null, llm: judge);
    }

    /// <summary>
    /// Один ход по тексту запроса
    /// </summary>
    /// <param name="prompt">Текст запроса</param>
    /// <param name="required">Требования к возможностям, которых нет в задании</param>
    /// <param name="weights">Профиль весов на этот ход; пусто, тогда профиль роутера</param>
    /// <param name="bar">Планка на этот ход, обязательная вероятность лайка; пусто, тогда планка роутера</param>
    /// <param name="cancellationToken">Токен отмены хода</param>
    public Task<RouterAnswer> AskAsync(string prompt, Capability required = Capability.None, RouteWeights? weights = null, double? bar = null, CancellationToken cancellationToken = default) =>
        AskAsync([new LLMMessage(LLMMessage.UserRole, prompt)], required, weights, bar, cancellationToken);

    /// <summary>
    /// Один ход по диалогу: задача распознается по последнему сообщению пользователя, а
    /// исполнителю уходит весь диалог целиком. То же, что ask_messages в версии на Python.
    /// </summary>
    /// <remarks>
    /// Объем входа для цены и окна контекста считается по всему диалогу: исполнитель читает его весь.
    /// Отмена токена прерывает ход; таймаут или сбой исполнителя передает работу запасному.
    /// </remarks>
    /// <param name="messages">Реплики диалога по порядку</param>
    /// <param name="required">Требования к возможностям, которых нет в задании</param>
    /// <param name="weights">Профиль весов на этот ход; пусто, тогда профиль роутера</param>
    /// <param name="bar">Планка на этот ход, обязательная вероятность лайка; пусто, тогда планка роутера</param>
    /// <param name="cancellationToken">Токен отмены хода</param>
    public async Task<RouterAnswer> AskAsync(IEnumerable<LLMMessage> messages, Capability required = Capability.None, RouteWeights? weights = null, double? bar = null, CancellationToken cancellationToken = default)
    {
        LLMMessage[] dialog = [.. messages];
        LLMMessage[] asked = [.. dialog.Where(message => string.Equals(message.Role, LLMMessage.UserRole, StringComparison.OrdinalIgnoreCase))];

        // Задание распознается по тексту, поэтому картинка или иное содержимое без текста задачей не считается
        string prompt = asked.LastOrDefault()?.Content as string ?? "";

        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("В диалоге нет сообщения пользователя, задачу распознать не из чего.", nameof(messages));

        double inputTokens = dialog.Sum(message => (message.Content as string)?.Length ?? 0) / InputFeaturesService.EST_SYMBOL_PER_TOKEN;
        Tracert trace = await Env.RouteAsync(prompt, Available(), _topk, required, WeightsFor(weights), asked.Length,
            SufficiencyBarFor(bar), _recognizer, inputTokens, _random, cancellationToken).ConfigureAwait(false);

        string text;

        try
        {
            text = await Env.ExecuteAsync(trace, candidate => _execute(candidate, dialog, cancellationToken), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            foreach (string failed in trace.Failed)
                _failedUntil[failed] = DateTimeOffset.UtcNow + FailureCooldown;
        }

        Specifications? actual = null;
        double? score = null;
        DiffSpec? critic = null;
        ContentReview? content = null;
        double? assessment = null;

        if (_measure && trace.RequestedSpec is not null)
        {
            try
            {
                // Форма и содержание не зависят друг от друга: замер структуры и суд содержания идут разом
                Task<Specifications> measuring = _measurer.GetSpecificationsAsync(text, cancellationToken);
                Task<ContentReview> reviewing = _contentJudge.ReviewAsync(prompt, trace.RequestedSpec, text, cancellationToken);

                actual = await measuring.ConfigureAwait(false);
                content = await reviewing.ConfigureAwait(false);
                score = Judge.Rate(trace, trace.RequestedSpec, actual);
                critic = Judge.Criticize(trace.RequestedSpec, actual, content);
                assessment = Judge.Assess(critic, content);
            }
            catch (Exception error) when (!cancellationToken.IsCancellationRequested)
            {
                // Сбой судьи (сеть, поставщик, таймаут, неполный вердикт) ход не роняет, ответ дороже оценки: ход идет
                // в журнал без автоотзыва. В журнал процесса только тип и код ошибки: текст исключения движка несет запрос
                System.Diagnostics.Trace.TraceWarning($"FAIRouter: ответ получен, но замерить его не удалось ({error.GetType().Name}{(ProviderRejectedException.Find(error) is { } rejected ? $", код {rejected.Code}" : "")}).");
                (actual, content, score, critic, assessment) = (null, null, null, null, null);
            }
        }

        long? roundId = _memory?.Append(trace, actual, prompt, assessment);

        return new RouterAnswer(text, trace.Winner, trace, trace.RequestedSpec, actual, score, critic, roundId, content, assessment, trace.BarReached);
    }

    /// <summary>
    /// Отзыв человека на ход: число от 0 до 1. Единица означает отличный ответ, ноль никуда не годный,
    /// 0,5 так себе. Отзыв перезаписывает автоотзыв судьи, на нем учатся и роутер, и судья; ход снова
    /// попадает в очередь обучения.
    /// </summary>
    /// <param name="roundId">Номер хода из RouterAnswer</param>
    /// <param name="score">Оценка от нуля (плохо) до единицы (отлично)</param>
    /// <param name="human">Человеческий отзыв или автоматический</param>
    public void Feedback(long roundId, double score, bool human = true)
    {
        if (score is < 0 or > 1 || double.IsNaN(score))
            throw new ArgumentOutOfRangeException(nameof(score), score, "Отзыв вне диапазона: нужно число от 0 (плохо) до 1 (отлично).");

        Memory().Feedback(roundId, score, human);
    }

    /// <summary>
    /// Веса хода: названные для хода, иначе веса роутера, иначе общие из Settings. Свой множитель
    /// температуры роутера подставляется в веса без своего множителя (веса по умолчанию и готовые
    /// профили Quality, Balance, Price); веса с заданным множителем берутся как есть.
    /// </summary>
    /// <param name="weights">Веса этого хода; пусто, тогда веса роутера</param>
    public RouteWeights? WeightsFor(RouteWeights? weights = null)
    {
        RouteWeights? chosen = weights ?? _weights;

        if (_temperatureScale is null)
            return chosen;

        RouteWeights resolved = chosen ?? Settings.Current;

        return resolved.TemperatureScale is null ? resolved with { TemperatureScale = _temperatureScale } : resolved;
    }

    /// <summary>
    /// Планка для хода по уровню: калибровка подбирается по журналу человеческих отзывов. Готовая
    /// <see cref="Bar"/> берется как есть. Пока отзывов меньше minRatings, планки нет и ход идет по
    /// метрике R: калибровать не на чем, а планка без калибровки отсекала бы наугад.
    /// </summary>
    /// <remarks>
    /// Калибровка запоминается и пересчитывается только после человеческого отзыва, обучения или
    /// загрузки: прежде каждый ход читал тысячу строк журнала и заново решал задачу Ньютона.
    /// </remarks>
    /// <param name="level">Обязательная вероятность лайка; пусто, тогда уровень роутера</param>
    public SufficiencyBar? SufficiencyBarFor(double? level = null)
    {
        if (Bar is not null)
            return Bar;

        double? bar = level ?? _bar;

        if (bar is null)
            return null;

        if (bar is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(level), bar, "Планка вне диапазона: нужна вероятность лайка от 0 до 1.");

        if (_memory is null)
            throw new InvalidOperationException("Планка по уровню требует журнала: создайте роутер с databasePath либо задайте готовую Bar с калибровкой.");

        (Calibration fit, double rate, int count) = _memory.CurrentCalibration();

        return count < _minRatings ? null : new SufficiencyBar(bar.Value, fit, rate);
    }

    /// <summary>
    /// Пары для калибровки планки из журнала: прогноз качества победителя в момент выбора и оценка
    /// человека, последние тысяча человеческих отзывов. Автоотзывы не берутся: планка обещает
    /// вероятность лайка человека, а не согласие судьи с самим собой.
    /// </summary>
    public IReadOnlyList<(double Quality, double Score)> CalibrationPairs() => Memory().CalibrationPairs();

    /// <summary>
    /// Обучение по ходам журнала, которые еще не учили, в порядке записи. Возвращает ошибку
    /// последней эпохи раздельно у роутера и судьи; к double приводится их суммой.
    /// </summary>
    /// <remarks>
    /// Каждый ход учит один раз: прежде каждый вызов заново проходил тысячу последних ходов от уже
    /// обученных весов, и многократный Train переобучал на них же. Ход считается обученным после
    /// <see cref="Save"/>: без сохранения веса пропали бы, а ход остался бы отмеченным. Новый отзыв
    /// возвращает ход в очередь.
    /// </remarks>
    /// <param name="epochs">Сколько раз пройти по новым ходам</param>
    public TrainingLoss Train(int epochs = 1) => Memory().Train(epochs);

    /// <summary>
    /// Сохраняет одной транзакцией векторы, которых касалось обучение (вместе с их средним задач), и
    /// матрицу судьи, затем отмечает обученные ходы. Необученные векторы не пишутся: при загрузке они
    /// затирали бы начальные веса по свежему снимку рейтингов.
    /// </summary>
    public void Save() => Memory().Save();

    /// <summary>
    /// Восстанавливает веса со средним задач, матрицу судьи и опыт кандидатов. Вектор чужой
    /// размерности пропускается с предупреждением, остальные загружаются.
    /// </summary>
    public void Load() => Memory().Load();

    // Кандидаты без недавнего отказа; отказали все, тогда все: пауза не повод остаться без исполнителя
    private BaseRoutedElement[] Available()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        BaseRoutedElement[] healthy = [.. Candidates.Where(candidate =>
            candidate.Name is null || !_failedUntil.TryGetValue(candidate.Name, out DateTimeOffset until) || until <= now)];

        return healthy.Length > 0 ? healthy : [.. Candidates];
    }

    private RouterMemory Memory() =>
        _memory ?? throw new InvalidOperationException("Журнал и хранилище не заданы: создайте роутер с databasePath.");
}
