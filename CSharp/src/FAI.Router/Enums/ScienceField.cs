namespace FAI.Router.Enums;

/// <summary>
/// Область науки, если задача научная. <see cref="None"/> означает, что задача не о науке.
/// </summary>
/// <remarks>
/// Уточняет <see cref="Domain"/>: область «наука» слишком широка, чтобы выучить, кто силен в
/// физике, а кто в экономике. Порядок значений задает разряды кода «один из многих».
/// </remarks>
public enum ScienceField
{
    None,
    Mathematics,
    Physics,
    Chemistry,
    Biology,
    Medicine,
    Economics,
    ComputerScience,
    SocialSciences,
    Humanities,
    Other
}
