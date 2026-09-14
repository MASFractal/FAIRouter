namespace FAI.Router.Enums;

/// <summary>
/// Язык программирования в задаче. <see cref="None"/> означает, что кода не заказано, а
/// <see cref="Other"/> означает код на языке вне списка.
/// </summary>
/// <remarks>
/// В готовом ответе язык считается по подписи ограждения кода (например, ` ```python`), в заказе
/// его распознает модель. Порядок значений задает разряды кода «один из многих».
/// </remarks>
public enum ProgrammingLanguage
{
    None,
    Python,
    JavaScript,
    TypeScript,
    CSharp,
    Java,
    Go,
    Rust,
    Cpp,
    Sql,
    Html,
    Other
}
