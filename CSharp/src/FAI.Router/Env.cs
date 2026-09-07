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
    /// Баллы за задачу в трассировке проставляет судья — после того, как победитель ответит.
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
        List<(double Score, BaseRoutedElement Element)> best = GetTopK(features, candidates, topk);

        return new Tracert
        {
            Winner = best[0].Element,
            TopKElements = [.. best.Select(item => item.Element)],
            InputFeatureVector = features.FeatureVector
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
