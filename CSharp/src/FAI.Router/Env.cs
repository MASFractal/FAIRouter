using FAI.Router.RotationTracking;
using FAI.Router.Enums;
using FAI.Router.RoutedElements;
using FAI.Router.Services;

namespace FAI.Router;


/// <summary>Итог выбора с планкой достаточности.</summary>
/// <param name="Top">
/// Лучшие кандидаты: прошедшие планку, упорядоченные по метрике R, либо, если не прошел никто,
/// упорядоченные по вероятности достаточности
/// </param>
/// <param name="Reached">Дотянул ли до планки хоть кто-нибудь</param>
public sealed record SufficientTop(List<(double Score, BaseRoutedElement Element)> Top, bool Reached);

/// <summary>
/// Среда для соревнования объектов роутинга
/// </summary>
public static class Env 
{
    /// <summary>
    /// Полный ход роутинга: признаки запроса, соревнование кандидатов, трассировка результата.
    /// Баллы за задачу в трассировке проставляет судья, после того как победитель ответит.
    /// </summary>
    /// <param name="textPrompt">Текст запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="topk">Сколько лучших оставить</param>
    /// <param name="required">Требования к возможностям, которых нет в спецификации</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    public static async Task<Tracert> RouteAsync(string textPrompt, IEnumerable<BaseRoutedElement> elements, int topk = 5, Capability required = Capability.None, RouteWeights? weights = null)
    {
        BaseRoutedElement[] candidates = [.. elements];

        // Проверка до обращения к модели: распознавание ТЗ стоит денег, а результат некому отдать
        if (candidates.Length == 0)
            throw new ArgumentException(
                "Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям.",
                nameof(elements));

        InputFeatures features = await InputFeaturesService.GetFeaturesAsync(textPrompt).ConfigureAwait(false);

        return Choose(features, candidates, topk, required, weights);
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
    public static Tracert Choose(InputFeatures features, IEnumerable<BaseRoutedElement> elements, int topk = 5, Capability required = Capability.None, RouteWeights? weights = null)
    {
        List<(double Score, BaseRoutedElement Element)> best = GetTopK(features, elements, topk, required, weights);

        if (best.Count == 0)
            throw new ArgumentException(
                "Ни один кандидат не подходит: список пуст либо все отсеяны по возможностям.",
                nameof(elements));

        int chosen = Sample(best, weights);

        return new Tracert
        {
            Winner = best[chosen].Element,
            TopKElements = [.. best.Select(item => item.Element)],
            InputFeatureVector = features.FeatureVector,
            RequestedSpec = features.InputSpecifications,
            IsExploration = chosen != 0
        };
    }

    /// <summary>
    /// Температура выбора для группы: T = C/K * сумма корней из D*_k / m_k.
    /// Кандидат, о котором мало данных, поднимает температуру всей группы, и группа начинает
    /// пробовать соперников вместо того, чтобы держаться за лидера.
    /// </summary>
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

            sum += Math.Sqrt(variance / Math.Max(element.Experience, 1));
            count++;
        }

        return count == 0 ? 0 : (weights ?? Settings.Current).TemperatureScale * sum / count;
    }

    // Выбор кандидата сэмплированием из softmax по оценкам топ-K
    private static int Sample(List<(double Score, BaseRoutedElement Element)> best, RouteWeights? weights)
    {
        double temperature = Temperature(best.Select(item => item.Element), weights);

        // Температура ушла в ноль: группа изучена и разброса в отзывах нет, брать лучшего
        if (temperature < 1e-9)
            return 0;

        // Оценки сдвигаются на лучшую из них: при малой температуре показатель степени
        // иначе улетает в бесконечность, и распределение обращается в NaN
        double top = best[0].Score;
        double[] chances = [.. best.Select(item => Math.Exp((item.Score - top) / temperature))];
        double total = chances.Sum();
        double dice = Random.Shared.NextDouble() * total;

        for (int i = 0; i < chances.Length; i++)
        {
            dice -= chances[i];

            if (dice <= 0)
                return i;
        }

        return 0;
    }

    /// <summary>
    /// Возвращает topK лучших элементов по метрике R 
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="topk">Сколько лучших оставить</param>
    /// <param name="required">Требования к возможностям, которых нет в спецификации</param>
    /// <param name="weights">Веса этого выбора; пусто, тогда берутся общие из Settings</param>
    public static List<(double Score, BaseRoutedElement Element)> GetTopK(InputFeatures features, IEnumerable<BaseRoutedElement> elements, int topk = 5, Capability required = Capability.None, RouteWeights? weights = null)
    {
        RouteWeights w = weights ?? Settings.Current;

        // Отсев по возможностям идет до сравнения оценок. Иначе кандидат, который заведомо не
        // справится, выигрывает по цене и скорости: метрика R о возможностях ничего не знает.
        BaseRoutedElement[] fit = [.. elements.Where(element => element.Supports(features.InputSpecifications, required))];

        if (fit.Length == 0)
            return [];

        // Метрика R считается относительно группы, а не по одному кандидату. Качество, цена и
        // время живут в несоизмеримых единицах: прогноз качества имеет порядок единицы, цена
        // запроса порядок 0,0002 доллара, и в прежней формуле вклад цены был в 1156 раз меньше
        // вклада качества. Роутер выбирал по качеству и времени, не глядя на деньги, и проигрывал
        // стратегии «всегда самая дешевая» вчетверо по качеству на единицу цены. Приведение
        // каждого слагаемого к нулевому среднему и единичному разбросу внутри группы делает веса
        // WQ, WC и Wt тем, чем они задуманы: долями важности сравнимых величин.
        double[] quality = Standardize([.. fit.Select(element => element.GetQualityScore(features.FeatureVector))]);
        double[] cost = Standardize([.. fit.Select(element => Math.Log(element.GetCost(features) + CostFloor))]);
        double[] time = Standardize([.. fit.Select(element => Math.Log(element.GetTime(features) + 2))]);

        // Слагаемые уже стандартизованы, поэтому масштаб суммы задают только веса: деление на
        // длину вектора весов делает R безразмерной величиной, а не зависящей от того, как
        // именно заданы WQ, WC и Wt. Без этого температура выбора была откалибрована под один
        // конкретный набор весов и требовала перекалибровки при любом заметном их изменении.
        double weightNorm = Math.Sqrt(w.WQ * w.WQ + w.WC * w.WC + w.Wt * w.Wt);
        double normalizer = weightNorm > 1e-12 ? weightNorm : 1.0;

        List<(double Score, BaseRoutedElement Element)> rElements = [.. fit.Select((element, i) =>
            ((w.WQ * quality[i] - w.WC * cost[i] - w.Wt * time[i]) / normalizer, element))];

        rElements.Sort((x, y) => -x.Score.CompareTo(y.Score)); // sort

        return rElements.Count > topk ? rElements[..topk] : rElements;
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
        var vector = features.FeatureVector;

        (double Sufficiency, BaseRoutedElement Element)[] scored = [.. elements
            .Where(element => element.Supports(features.InputSpecifications, required))
            .Select(element => (bar.Sufficiency(element.Experience, element.GetQualityScore(vector)), element))];

        BaseRoutedElement[] passing = [.. scored.Where(item => item.Sufficiency >= bar.Bar).Select(item => item.Element)];

        if (passing.Length > 0)
        {
            RouteWeights w = weights ?? Settings.Current;
            RouteWeights floored = w with { WQ = SufficientQualityShare * (w.WC + w.Wt) };

            return new SufficientTop(GetTopK(features, passing, topk, required, floored), Reached: true);
        }

        return new SufficientTop([.. scored.OrderByDescending(item => item.Sufficiency).Take(topk)], Reached: false);
    }

    /// <summary>
    /// Нижняя граница цены под логарифмом, доллары. Бесплатные модели дают нулевую стоимость,
    /// а логарифм нуля ушел бы в бесконечность и задавил бы разброс остальных кандидатов.
    /// </summary>
    private const double CostFloor = 1e-5;

    /// <summary>
    /// Вес качества среди прошедших планку, в долях от суммы весов цены и времени. Качество им уже
    /// обеспечено, и платить за лишнее незачем: решают цена и время, а качество остается только
    /// разнимать равных.
    /// </summary>
    /// <remarks>
    /// Доля, а не число. Постоянный вес 0,05 у заказчика, которому цена почти безразлична (вес цены
    /// тоже 0,05), уравнивал качество с ценой, и сильная модель снова выигрывала у достаточной.
    /// </remarks>
    private const double SufficientQualityShare = 0.1;

    // Приведение к нулевому среднему и единичному разбросу внутри группы. Одинаковые у всех
    // значения дают нули: такое слагаемое на выбор не влияет, и это верно
    private static double[] Standardize(double[] values)
    {
        if (values.Length < 2)
            return new double[values.Length];

        double mean = values.Average();
        double deviation = Math.Sqrt(values.Sum(value => (value - mean) * (value - mean)) / values.Length);

        return deviation < 1e-12
            ? new double[values.Length]
            : [.. values.Select(value => (value - mean) / deviation)];
    }

    /// <summary>
    /// Проводит ход с запасными вариантами: если победитель отказал, работа переходит следующему
    /// кандидату из топ-K. Победитель в трассировке заменяется на того, кто справился.
    /// </summary>
    /// <remarks>
    /// Замена победителя обязательна. Обучение поощряет того, кто записан победителем, и без
    /// замены похвалу получил бы кандидат, который ничего не сделал, а сделавший работу остался
    /// бы ни с чем.
    /// </remarks>
    /// <param name="trace">Трассировка хода</param>
    /// <param name="run">Как выполнить работу выбранным кандидатом</param>
    public static async Task<TResult> ExecuteAsync<TResult>(Tracert trace, Func<BaseRoutedElement, Task<TResult>> run)
    {
        BaseRoutedElement[] chain = [trace.Winner, .. trace.TopKElements.Where(element => element != trace.Winner)];
        List<Exception> failures = [];

        foreach (BaseRoutedElement candidate in chain)
        {
            try
            {
                TResult result = await run(candidate).ConfigureAwait(false);
                trace.Winner = candidate;

                return result;
            }
            // Отмена запасным вариантом не является: отмененный запрос уходил к следующему
            // кандидату, потом к следующему, и вместо тихого выхода получался AggregateException
            catch (Exception exception) when (exception is not OperationCanceledException)
            {
                failures.Add(exception);
            }
        }

        throw new AggregateException(
            $"Отказали все {chain.Length} кандидатов из топ-K, ход выполнить некому.", failures);
    }
}
