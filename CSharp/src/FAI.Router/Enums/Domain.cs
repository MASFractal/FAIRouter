namespace FAI.Router.Enums;

/// <summary>
/// Предметная область задачи. Первое значение означает «не задана», и в векторе признаков она
/// не светится.
/// </summary>
/// <remarks>
/// Первые девять областей повторяют профессиональные категории арены (arena.ai), следующие шесть
/// взяты там, где различать модели позволяют данные Artificial Analysis: бизнес-функции
/// AutomationBench (финансы, продажи, кадры, поддержка, операции) и индекс инженерии. Business
/// остается управлением и стратегией. Порядок значений задает разряды кода «один из многих»,
/// переставлять их нельзя: сохраненные веса перестанут соответствовать координатам.
/// </remarks>
public enum Domain
{
    General,
    SoftwareIt,
    Writing,
    Science,
    Entertainment,
    Business,
    Math,
    Legal,
    Medicine,
    Marketing,
    Finance,
    Sales,
    Hr,
    CustomerSupport,
    Operations,
    Engineering
}
