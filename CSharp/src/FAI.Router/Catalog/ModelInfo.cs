using FAI.Router.Enums;

namespace FAI.Router.Catalog;

/// <summary>
/// Сведения о модели из каталога поставщика
/// </summary>
/// <param name="Id">Опознавательное имя модели у поставщика</param>
/// <param name="Title">Человекочитаемое название</param>
/// <param name="DollarsPerMillionInput">Цена за миллион входных токенов</param>
/// <param name="DollarsPerMillionOutput">Цена за миллион выходных токенов</param>
/// <param name="ContextTokens">Размер окна в токенах</param>
/// <param name="MaxAnswerTokens">Наибольший ответ в токенах; ноль означает, что поставщик не указал</param>
/// <param name="Capabilities">Что модель умеет, насколько это видно из каталога</param>
public record ModelInfo(
    string Id,
    string Title,
    double DollarsPerMillionInput,
    double DollarsPerMillionOutput,
    int ContextTokens,
    int MaxAnswerTokens,
    Capability Capabilities);
