using AI.DataStructs.Algebraic;
using FAI.Router.RotationTracking;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.RoutedElements;
using FAI.Router.Services;

namespace FAI.Router;

/// <summary>Итог выбора с планкой достаточности.</summary>
/// <param name="Top">
/// Лучшие кандидаты: прошедшие планку, упорядоченные по метрике R, либо, если не прошел никто,
/// упорядоченные по вероятности достаточности (при равной по прогнозу качества, затем по имени)
/// </param>
/// <param name="Reached">Дотянул ли до планки хоть кто-нибудь</param>
public sealed record SufficientTop(List<(double Score, BaseRoutedElement Element)> Top, bool Reached);

/// <summary>
/// Среда для соревнования объектов роутинга
/// </summary>
public static class Env
{
    /// <summary>
    /// Вес качества среди прошедших планку, в долях от суммы весов цены и времени. Качество им уже
    /// обеспечено, и платить за лишнее незачем: решают цена и время, а качество остается только
    /// разнимать равных.
    /// </summary>
    /// <remarks>
    /// Доля, а не число. Постоянный вес 0,05 у заказчика, которому цена почти безразлична (вес цены
    /// тоже 0,05), уравнивал качество с ценой, и сильная модель снова выигрывала у достаточной. У
    /// профиля «только качество» (цена и время по нулю) доля дала бы ноль, и порядок задавал бы
    /// каталог; там качество получает вес единицу.
    /// </remarks>
    private const double SufficientQualityShare = 0.1;

    /// <summary>
    /// Нижняя граница цены под логарифмом, доля самой низкой положительной цены группы. Бесплатные
    /// модели дают нулевую стоимость, а логарифм нуля ушел бы в бесконечность. Граница относительная:
    /// прежняя абсолютная 1e-5 доллара значила разное для рублевого каталога и для хода на тысячу
    /// токенов. Бесплатная модель выходит вдесятеро дешевле самой дешевой платной.
    /// </summary>
    private const double CostFloorShare = 0.1;

    /// <summary>
    /// Нижняя граница разброса логарифма цены и времени при стандартизации. Без нее группа, где
    /// цены различаются на проценты, раздувала эти проценты до единичного разброса, и копеечная
    /// разница решала ход наравне с десятикратной.
    /// </summary>
    private const double MinLogDeviation = 0.25;

    /// <summary>Нижняя граница разброса прогноза качества при стандартизации, по той же причине</summary>
    private const double MinQualityDeviation = 0.05;

    /// <summary>
    /// Полный ход роутинга: признаки запроса, соревнование кандидатов, трассировка результата.
    /// Баллы за задачу в трассировке проставляет судья, после того как победитель ответит.
    /// </summary>
    /// <param name="textPrompt">Текст запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="topk">Сколько лучших оставить</param>
    /// <param name="required">Требования к возможностям, которых нет в спецификации</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    /// <param name="turns">Сколько реплик пользователя в диалоге</param>
    /// <param name="bar">Планка достаточности; пусто, тогда выбор по метрике R</param>
    /// <param name="specs">Кто распознает задание; пусто, значит общий распознаватель через Settings.LLM</param>
    /// <param name="inputTokens">Объем входа всего диалога в токенах; пусто, значит по тексту запроса</param>
    /// <param name="random">Генератор жребия; пусто, значит общий</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public static async Task<Tracert> RouteAsync(
        string textPrompt,
        IEnumerable<BaseRoutedElement> elements,
        int topk = 5,
        Capability required = Capability.None,
        RouteWeights? weights = null,
        int turns = 1,
        SufficiencyBar? bar = null,
        ISpecService? specs = null,
        double? inputTokens = null,
        Random? random = null,
        CancellationToken cancellationToken = default)
    {
        BaseRoutedElement[] candidates = [.. elements];

        // Проверка до обращения к модели: распознавание ТЗ стоит денег, а результат некому отдать
        if (candidates.Length == 0)
            throw new ArgumentException(
                "Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям.",
                nameof(elements));

        cancellationToken.ThrowIfCancellationRequested();

        // Сбой распознавания хода не роняет: выбор идет по типовой задаче, а заказа в трассировке нет,
        // и замер ответа пропускается, потому что сверять его не с чем
        InputFeatures features = InputFeaturesService.GetFeatures(textPrompt);
        Specifications? recognized = await InputFeaturesService.RecognizeAsync(textPrompt, specs, cancellationToken).ConfigureAwait(false);

        if (recognized is not null)
            InputFeaturesService.Apply(features, recognized);

        features.TurnCount = Math.Max(turns, 1);

        if (inputTokens is > 0)
            features.InputLen = inputTokens.Value;

        Tracert trace = bar is null
            ? Choose(features, candidates, topk, required, weights, random)
            : ChooseSufficient(features, candidates, bar.Value, topk, required, weights, random);
        trace.RequestedSpec = recognized;

        return trace;
    }

    /// <summary>
    /// Выбор исполнителя по готовым признакам, без обращения к модели.
    /// Ход достается не обязательно лучшему: из оценок топ-K строится распределение через softmax
    /// с температурой, и кандидат берется сэмплированием. Температура падает по мере накопления
    /// опыта, поэтому изученная группа выбирает почти жадно, а группа с новичками пробует их чаще.
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="topk">Сколько лучших оставить</param>
    /// <param name="required">Требования к возможностям, которых нет в спецификации</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    /// <param name="random">Генератор жребия; пусто, значит общий</param>
    public static Tracert Choose(InputFeatures features, IEnumerable<BaseRoutedElement> elements, int topk = 5, Capability required = Capability.None, RouteWeights? weights = null, Random? random = null)
    {
        List<(double Score, BaseRoutedElement Element)> best = GetTopK(features, elements, topk, required, weights);

        if (best.Count == 0)
            throw new ArgumentException(
                "Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям.",
                nameof(elements));

        return Trace(features, best, Sample(best, weights, random), required, reached: null);
    }

    /// <summary>
    /// Выбор с планкой достаточности, оформленный трассировкой: среди дотянувших до планки ход
    /// разыгрывается как в <see cref="Choose"/>, по метрике R с температурой. Не дотянул никто, тогда
    /// жребий идет среди сильнейших по вероятности достаточности: без него при недоборе ход всегда
    /// доставался первому, остальные не набирали опыта, и выйти из недобора было не на чем.
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="bar">Планка достаточности этого выбора</param>
    /// <param name="topk">Сколько лучших оставить</param>
    /// <param name="required">Требования к возможностям, которых нет в спецификации</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    /// <param name="random">Генератор жребия; пусто, значит общий</param>
    public static Tracert ChooseSufficient(InputFeatures features, IEnumerable<BaseRoutedElement> elements, SufficiencyBar bar, int topk = 5, Capability required = Capability.None, RouteWeights? weights = null, Random? random = null)
    {
        SufficientTop chosen = GetSufficient(features, elements, bar, topk, required, weights);

        if (chosen.Top.Count == 0)
            throw new ArgumentException(
                "Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям.",
                nameof(elements));

        return Trace(features, chosen.Top, Sample(chosen.Top, weights, random), required, chosen.Reached);
    }

    /// <summary>
    /// Проводит ход с запасными вариантами: если победитель отказал, работа переходит следующему
    /// кандидату из топ-K. Победитель в трассировке заменяется на того, кто справился.
    /// </summary>
    /// <remarks>
    /// Замена победителя обязательна. Обучение поощряет того, кто записан победителем, и без
    /// замены похвалу получил бы кандидат, который ничего не сделал, а сделавший работу остался
    /// бы ни с чем. Пустой текст считается отказом. Отказавшие записываются в <see cref="Tracert.Failed"/>.
    /// <para>
    /// Отменой считается только отмена токена вызывающего. Таймаут клиента тоже приходит
    /// исключением отмены (TaskCanceledException), и прежде он ронял весь ход вместо того, чтобы
    /// отдать работу запасному.
    /// </para>
    /// </remarks>
    /// <param name="trace">Трассировка хода</param>
    /// <param name="run">Как выполнить работу выбранным кандидатом</param>
    /// <param name="cancellationToken">Токен отмены вызывающего</param>
    public static async Task<TResult> ExecuteAsync<TResult>(Tracert trace, Func<BaseRoutedElement, Task<TResult>> run, CancellationToken cancellationToken = default)
    {
        BaseRoutedElement[] chain = [trace.Winner, .. trace.TopKElements.Where(element => element != trace.Winner)];
        List<Exception> failures = [];

        foreach (BaseRoutedElement candidate in chain)
        {
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                TResult result = await run(candidate).ConfigureAwait(false);

                if (result is string text && string.IsNullOrWhiteSpace(text))
                    throw new InvalidOperationException($"Кандидат {candidate.Name} вернул пустой ответ.");

                if (candidate != trace.Winner)
                {
                    trace.Winner = candidate;
                    trace.IsExploration = trace.TopKElements.Count > 0 && candidate != trace.TopKElements[0];
                    trace.Forecast = candidate.GetQualityScore(trace.InputFeatureVector);
                }

                return result;
            }
            catch (Exception exception) when (!cancellationToken.IsCancellationRequested)
            {
                failures.Add(exception);
                trace.Failed.Add(candidate.Name ?? "");
            }
        }

        throw new AggregateException(
            $"Отказали все {chain.Length} кандидатов из топ-K, ход выполнить некому.", failures);
    }

    /// <summary>
    /// Возвращает topK лучших элементов по метрике R. Ничьи разбираются по прогнозу качества, затем
    /// по имени, поэтому порядок не зависит от порядка каталога.
    /// </summary>
    /// <remarks>
    /// Объем хода не вошел ни в одного кандидата с нужными возможностями: отдается один кандидат с
    /// наибольшим пределом, и <see cref="Tracert.ContextShortfall"/> у выбора это отмечает. Прежде пул
    /// пустел, и ход падал исключением.
    /// </remarks>
    /// <param name="features">Признаки запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="topk">Сколько лучших оставить</param>
    /// <param name="required">Требования к возможностям, которых нет в спецификации</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    public static List<(double Score, BaseRoutedElement Element)> GetTopK(InputFeatures features, IEnumerable<BaseRoutedElement> elements, int topk = 5, Capability required = Capability.None, RouteWeights? weights = null)
    {
        Vector vector = features.FeatureVector;
        BaseRoutedElement[] fit = Fit(features, elements, required);

        return Rank(features, fit, Quality(fit, vector), topk, weights ?? Settings.Current);
    }

    /// <summary>
    /// Выбор по принципу «необходимо и достаточно»: отсеять тех, кто прогнозируемо не дотягивает до
    /// планки, а среди остальных взять дешевого и быстрого.
    /// </summary>
    /// <remarks>
    /// Не прошел никто, тогда отдаются сильнейшие по вероятности достаточности, а <c>Reached</c>
    /// ложно. Поднимать ли цену или предупредить человека, решает вызывающий: у него есть то, чего
    /// нет у библиотеки, то есть сам человек.
    /// </remarks>
    /// <param name="features">Признаки запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="bar">Планка достаточности этого выбора</param>
    /// <param name="topk">Сколько лучших оставить</param>
    /// <param name="required">Требования к возможностям, которых нет в спецификации</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    public static SufficientTop GetSufficient(InputFeatures features, IEnumerable<BaseRoutedElement> elements, SufficiencyBar bar, int topk = 5, Capability required = Capability.None, RouteWeights? weights = null)
    {
        BaseRoutedElement[] fit = Fit(features, elements, required);
        double[] quality = Quality(fit, features.FeatureVector);
        double[] sufficiency = [.. fit.Select((element, i) => bar.Sufficiency(element.Experience, quality[i]))];
        int[] passing = [.. Enumerable.Range(0, fit.Length).Where(i => sufficiency[i] >= bar.Bar)];

        if (passing.Length > 0)
        {
            RouteWeights w = weights ?? Settings.Current;
            double share = SufficientQualityShare * (w.WC + w.Wt);
            RouteWeights floored = w with { WQ = share > 1e-12 ? share : 1 };

            return new SufficientTop(
                Rank(features, [.. passing.Select(i => fit[i])], [.. passing.Select(i => quality[i])], topk, floored), Reached: true);
        }

        int[] order = [.. Enumerable.Range(0, fit.Length)
            .OrderByDescending(i => sufficiency[i])
            .ThenByDescending(i => quality[i])
            .ThenBy(i => fit[i].Name, StringComparer.Ordinal)];

        return new SufficientTop([.. order.Take(topk).Select(i => (sufficiency[i], fit[i]))], Reached: false);
    }

    /// <summary>
    /// Температура выбора для группы: T = C/K * сумма корней из D*_k / m_k.
    /// Кандидат, о котором мало данных, поднимает температуру всей группы, и группа начинает
    /// пробовать соперников вместо того, чтобы держаться за лидера.
    /// </summary>
    /// <remarks>
    /// m_k это условный опыт: отзывы людей, автоотзывы с весом и опыт, который стоят начальные веса
    /// по рейтингам (<see cref="BaseRoutedElement.PriorExperience"/>). Без автоотзывов и рейтингов
    /// разведка не остывала, пока не придут человеческие отзывы, а их бывает ноль.
    /// </remarks>
    /// <param name="group">Кандидаты, отобранные в топ-K</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    public static double Temperature(IEnumerable<BaseRoutedElement> group, RouteWeights? weights = null)
    {
        double sum = 0;
        int count = 0;

        foreach (BaseRoutedElement element in group)
        {
            // Дисперсия по одному ходу равна нулю и означала бы уверенность на пустом месте,
            // поэтому кандидат считается изученным начиная со второго оцененного хода
            bool known = element.Experience >= 2;
            double variance = known ? element.ScoreVariance : Settings.UnknownVariance;

            sum += Math.Sqrt(variance / Math.Max(element.Experience + element.PriorExperience, 1));
            count++;
        }

        return count == 0 ? 0 : (weights ?? Settings.Current).ResolvedTemperatureScale * sum / count;
    }

    /// <summary>
    /// Выбор кандидата сэмплированием из softmax по оценкам топ-K: индекс в списке, ноль при
    /// нулевой температуре. Открыт для вызывающих, которые строят топ-K сами (например, среди
    /// прошедших планку достаточности) и хотят ту же разведку, что у <see cref="Choose"/>.
    /// </summary>
    /// <param name="best">Кандидаты, упорядоченные по метрике R, лучший первым</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    /// <param name="random">Генератор жребия; пусто, значит общий. Чужой генератор берется под блокировку</param>
    public static int Sample(List<(double Score, BaseRoutedElement Element)> best, RouteWeights? weights = null, Random? random = null)
    {
        if (best.Count < 2)
            return 0;

        double temperature = Temperature(best.Select(item => item.Element), weights);

        // Температура ушла в ноль: группа изучена и разброса в отзывах нет, брать лучшего
        if (temperature < 1e-9)
            return 0;

        // Оценки сдвигаются на наибольшую из них: при малой температуре показатель степени
        // иначе улетает в бесконечность, и распределение обращается в NaN
        double top = best.Max(item => item.Score);
        double[] chances = [.. best.Select(item => Math.Exp((item.Score - top) / temperature))];
        double dice = Unit(random ?? Random.Shared) * chances.Sum();

        for (int i = 0; i < chances.Length; i++)
        {
            dice -= chances[i];

            if (dice <= 0)
                return i;
        }

        return 0;
    }

    private static Tracert Trace(InputFeatures features, List<(double Score, BaseRoutedElement Element)> top, int index, Capability required, bool? reached)
    {
        Vector vector = features.FeatureVector;
        BaseRoutedElement winner = top[index].Element;

        return new Tracert
        {
            Winner = winner,
            TopKElements = [.. top.Select(item => item.Element)],
            InputFeatureVector = vector,
            RequestedSpec = features.InputSpecifications,
            IsExploration = index != 0,
            Forecast = winner.GetQualityScore(vector),
            ContextShortfall = !winner.Supports(features, required),
            BarReached = reached
        };
    }

    // Отсев по возможностям и объему до сравнения оценок. Иначе кандидат, который заведомо не
    // справится, выигрывает по цене и скорости: метрика R о возможностях ничего не знает
    private static BaseRoutedElement[] Fit(InputFeatures features, IEnumerable<BaseRoutedElement> elements, Capability required)
    {
        BaseRoutedElement[] able = [.. elements.Where(element => element.Can(features.InputSpecifications, required))];
        BaseRoutedElement[] fit = [.. able.Where(element => element.Supports(features, required))];

        if (fit.Length > 0 || able.Length == 0)
            return fit;

        // Объем не вошел ни в кого: лучше ответ того, у кого предел больше, чем никакого
        return [able.MaxBy(element => (Room(element.ContextWindow), Room(element.ContextLimit)))!];
    }

    private static double Room(int limit) => limit > 0 ? limit : double.PositiveInfinity;

    private static double[] Quality(BaseRoutedElement[] fit, Vector vector) =>
        [.. fit.Select(element => element.GetQualityScore(vector))];

    // Метрика R относительно группы. Качество, цена и время живут в несоизмеримых единицах: прогноз
    // качества имеет порядок единицы, цена запроса порядок 0,0002 доллара, и в прежней формуле вклад
    // цены был в 1156 раз меньше вклада качества. Приведение каждого слагаемого к нулевому среднему
    // и единичному разбросу внутри группы делает веса WQ, WC и Wt долями важности сравнимых величин.
    // Деление на длину вектора весов делает R безразмерной: температура не зависит от того, как
    // именно заданы веса
    private static List<(double Score, BaseRoutedElement Element)> Rank(InputFeatures features, BaseRoutedElement[] fit, double[] quality, int topk, RouteWeights w)
    {
        double[] costs = [.. fit.Select(element => element.HasKnownPrice ? element.GetCost(features) : double.NaN)];
        double floor = CostFloor(costs);

        double[] q = Standardize(quality, lowIsWorst: true, MinQualityDeviation);
        double[] cost = Standardize([.. costs.Select(value => Math.Log(value + floor))], lowIsWorst: false, MinLogDeviation);
        double[] time = Standardize([.. fit.Select(element => element.TPS > 0 ? Math.Log(element.GetTime(features) + 2) : double.NaN)], lowIsWorst: false, MinLogDeviation);

        double weightNorm = Math.Sqrt(w.WQ * w.WQ + w.WC * w.WC + w.Wt * w.Wt);
        double normalizer = weightNorm > 1e-12 ? weightNorm : 1.0;

        double[] scores = [.. Enumerable.Range(0, fit.Length).Select(i => (w.WQ * q[i] - w.WC * cost[i] - w.Wt * time[i]) / normalizer)];

        return [.. Enumerable.Range(0, fit.Length)
            .OrderByDescending(i => scores[i])
            .ThenByDescending(i => double.IsFinite(quality[i]) ? quality[i] : double.MinValue)
            .ThenBy(i => fit[i].Name, StringComparer.Ordinal)
            .Take(topk)
            .Select(i => (scores[i], fit[i]))];
    }

    private static double CostFloor(double[] costs)
    {
        double[] positive = [.. costs.Where(cost => double.IsFinite(cost) && cost > 0)];

        return positive.Length == 0 ? 1 : CostFloorShare * positive.Min();
    }

    // Приведение к нулевому среднему и единичному разбросу внутри группы. Нечисловое значение
    // (неизвестная цена, NaN в векторе) ставится на шаг хуже худшего в группе, а не отравляет всю
    // группу и не равняется с худшим известным. Разброс не меньше заданной границы: близкие значения
    // не раздуваются до единичного разброса
    private static double[] Standardize(double[] values, bool lowIsWorst, double minDeviation)
    {
        double[] finite = [.. values.Where(double.IsFinite)];

        if (values.Length < 2 || finite.Length == 0)
            return new double[values.Length];

        double step = Math.Max(finite.Max() - finite.Min(), minDeviation);
        double worst = lowIsWorst ? finite.Min() - step : finite.Max() + step;
        double[] clean = [.. values.Select(value => double.IsFinite(value) ? value : worst)];
        double mean = clean.Average();
        double deviation = Math.Max(Math.Sqrt(clean.Sum(value => (value - mean) * (value - mean)) / clean.Length), minDeviation);

        return deviation < 1e-12
            ? new double[values.Length]
            : [.. clean.Select(value => (value - mean) / deviation)];
    }

    private static double Unit(Random random)
    {
        lock (random)
            return random.NextDouble();
    }
}
