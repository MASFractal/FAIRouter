namespace FAI.Router.Enums;

/// <summary>
/// Тип задачи: что заказчик хочет получить на выходе. Первое значение означает «не задан», и в
/// векторе признаков оно не светится.
/// </summary>
/// <remarks>
/// Виды собраны по бизнес-задачам пользователей и сгруппированы так, чтобы каждая группа ложилась
/// на категорию арены (arena.ai) или индекс Artificial Analysis: начальный прогноз у вида есть
/// без собственных замеров, а различие внутри категории (письмо против коммерческого предложения)
/// выучит журнал. Порядок значений задает разряды вектора, переставлять их нельзя.
/// </remarks>
public enum TaskKind
{
    None,

    // Письма и коммуникации
    BusinessLetter,
    CommercialOffer,
    CustomerReply,
    InternalMemo,
    EmailCampaign,

    // Отчеты и аналитика
    AnalyticalReport,
    FinancialReport,
    MarketResearch,
    DataAnalysis,
    Summary,
    MeetingMinutes,

    // Маркетинг и продажи
    MarketingArticle,
    SocialPost,
    AdCopy,
    ProductDescription,
    LandingPage,
    PressRelease,
    SalesScript,

    // Документы и право
    LegalDocument,
    LegalAnalysis,
    Policy,
    JobDescription,

    // Код и ИТ
    CodeWriting,
    CodeReview,
    TechnicalDoc,

    // Наука и обучение
    Explanation,
    ResearchReview,
    LearningMaterial,
    MathSolution,

    // Творчество и медиа
    Story,
    Script,

    // Консультации и планы
    Advice,
    BusinessPlan,

    Other
}
