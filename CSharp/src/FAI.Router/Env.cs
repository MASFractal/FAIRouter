using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using FAI.Router.Services;

namespace FAI.Router;


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
    public static async Task<Tracert> RouteAsync(string textPrompt, IEnumerable<BaseRoutedElement> elements, int topk = 5)
    {
        BaseRoutedElement[] candidates = [.. elements];

        // Проверка до обращения к модели: распознавание ТЗ стоит денег, а результат некому отдать
        if (candidates.Length == 0)
            throw new ArgumentException("Список кандидатов пуст.", nameof(elements));

        InputFeatures features = await InputFeaturesService.GetFeaturesAsync(textPrompt).ConfigureAwait(false);

        return Choose(features, candidates, topk);
    }

    /// <summary>
    /// Выбор исполнителя по готовым признакам, без обращения к модели.
    /// Обычно ход достается лучшему по метрике R, но с вероятностью Settings.ExplorationRate
    /// вместо него ход получает случайный соперник. Без такой разведки победитель первого хода
    /// закрепляет сам себя: соперники не получают ни одного хода и не могут показать, что
    /// где-то лучше, а журнал наполняется однородными данными.
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="topk">Сколько лучших оставить</param>
    public static Tracert Choose(InputFeatures features, IEnumerable<BaseRoutedElement> elements, int topk = 5)
    {
        List<(double Score, BaseRoutedElement Element)> best = GetTopK(features, elements, topk);

        if (best.Count == 0)
            throw new ArgumentException("Список кандидатов пуст.", nameof(elements));

        bool exploring = best.Count > 1 && Random.Shared.NextDouble() < Settings.ExplorationRate;

        // На разведке лидер исключается намеренно. Равномерный жребий по всему списку отдавал бы
        // ему еще и долю разведочных ходов, и соперники набирали бы опыт заметно медленнее.
        int chosen = exploring ? Random.Shared.Next(1, best.Count) : 0;

        return new Tracert
        {
            Winner = best[chosen].Element,
            TopKElements = [.. best.Select(item => item.Element)],
            InputFeatureVector = features.FeatureVector,
            RequestedSpec = features.InputSpecifications,
            IsExploration = exploring
        };
    }

    /// <summary>
    /// Возвращает topK лучших элементов по метрике R 
    /// </summary>
    /// <param name="features">Признаки запроса</param>
    /// <param name="elements">Кандидаты на исполнение</param>
    /// <param name="topk">Сколько лучших оставить</param>
    public static List<(double Score, BaseRoutedElement Element)> GetTopK(InputFeatures features, IEnumerable<BaseRoutedElement> elements, int topk = 5)
    {
        List<(double Score, BaseRoutedElement Element)> rElements =
            [.. elements.Select(element => (element.GetRScore(features), element))];

        rElements.Sort((x, y) => -x.Score.CompareTo(y.Score)); // sort

        return rElements.Count > topk ? rElements[..topk] : rElements;
    }
}
