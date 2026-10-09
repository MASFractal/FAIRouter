namespace FAI.Router.RoutedElements;

/// <summary>
/// Действия над набором кандидатов
/// </summary>
public static class RoutedElementExtensions
{
    /// <summary>
    /// Кандидаты по имени. Безымянные пропускаются, из повторов имени берется первый: повтор модели в
    /// списке раньше ронял журнал и обучение исключением о дубликате ключа.
    /// </summary>
    /// <param name="elements">Кандидаты</param>
    public static Dictionary<string, BaseRoutedElement> ByName(this IEnumerable<BaseRoutedElement> elements)
    {
        Dictionary<string, BaseRoutedElement> byName = [];

        foreach (BaseRoutedElement element in elements)
            if (!string.IsNullOrWhiteSpace(element.Name))
                byName.TryAdd(element.Name, element);

        return byName;
    }
}
