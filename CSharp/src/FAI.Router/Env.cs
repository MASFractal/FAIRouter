using FAI.Router.RoutedElements;
using FAI.Router.Services;

namespace FAI.Router;


/// <summary>
/// Среда для соревнования объектов роутинга
/// </summary>
public class Env 
{
    /// <summary>
    /// Возвращает topK лучших элементов по метрике R 
    /// </summary>
    public static List<(double, BaseRoutedElement)> GetTopK(string textPrompt, IEnumerable<BaseRoutedElement> elements, int topk = 5)
    {
        var features = InputFeaturesService.GetFeatures(textPrompt);
        List<(double, BaseRoutedElement)> rElements = new List<(double, BaseRoutedElement)>(elements.Count());

        foreach (var item in elements)
            rElements.Add(
                (item.GetRScore(features),
                item)
                );

        rElements.Sort((x, y) => -x.Item1.CompareTo(y.Item1)); // sort
        
        return rElements.Count > topk? rElements[..topk] : rElements;
    }
}
