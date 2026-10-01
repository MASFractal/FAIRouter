namespace FAI.Router.Enums;

/// <summary>
/// Что кандидат умеет. Возможности проверяются до сравнения оценок: кандидат, который
/// заведомо не справится, не должен побеждать по цене и скорости.
/// </summary>
[Flags]
public enum Capability
{
    /// <summary>
    /// Ничего сверх обычного текста
    /// </summary>
    None = 0,

    /// <summary>
    /// Умеет писать код
    /// </summary>
    Code = 1,

    /// <summary>
    /// Умеет формулы
    /// </summary>
    Formulas = 2,

    /// <summary>
    /// Понимает изображения
    /// </summary>
    Vision = 4,

    /// <summary>
    /// Умеет вызывать инструменты
    /// </summary>
    Tools = 8,

    /// <summary>
    /// Умеет искать в сети
    /// </summary>
    WebSearch = 16,

    /// <summary>
    /// Все перечисленное. Значение по умолчанию у кандидата: пока ограничения не заданы,
    /// он считается способным на все, и поведение прежнее.
    /// </summary>
    All = Code | Formulas | Vision | Tools | WebSearch
}
