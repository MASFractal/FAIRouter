from enum import Enum, IntEnum, IntFlag


class Indexed(Enum):
    """Перечисление с разрядом в векторе признаков: порядок значений задает разряды кода «один из
    многих», поэтому переставлять их нельзя, сохраненные веса перестанут соответствовать
    координатам. Значения совпадают с именами в версии на C#: так схемы ответа модели и журналы
    читаются обеими версиями одинаково."""

    @property
    def index(self) -> int:
        return list(type(self)).index(self)


class Style(Indexed):
    """Стиль текста."""

    JOURNALISTIC = "Journalistic"
    GOST = "GOST"
    CONVERSATIONAL = "Conversational"
    OFFICIAL_BUSINESS = "OfficialBusiness"
    SCIENTIFIC = "Scientific"
    LITERARY = "Literary"
    TECHNICAL = "Technical"
    ADVERTISING = "Advertising"
    CHILDREN = "Children"
    OTHER = "Other"


class Domain(Indexed):
    """Предметная область задачи. Первое значение означает «не задана», и в векторе она не
    светится. Первые девять областей повторяют профессиональные категории арены (arena.ai),
    следующие шесть взяты там, где различать модели позволяют данные Artificial Analysis:
    бизнес-функции AutomationBench и индекс инженерии. BUSINESS остается управлением и стратегией."""

    GENERAL = "General"
    SOFTWARE_IT = "SoftwareIt"
    WRITING = "Writing"
    SCIENCE = "Science"
    ENTERTAINMENT = "Entertainment"
    BUSINESS = "Business"
    MATH = "Math"
    LEGAL = "Legal"
    MEDICINE = "Medicine"
    MARKETING = "Marketing"
    FINANCE = "Finance"
    SALES = "Sales"
    HR = "Hr"
    CUSTOMER_SUPPORT = "CustomerSupport"
    OPERATIONS = "Operations"
    ENGINEERING = "Engineering"


class ProgrammingLanguage(Indexed):
    """Язык программирования в задаче. NONE означает, что кода не заказано, OTHER означает код
    на языке вне списка. В готовом ответе язык считается по подписи ограждения кода, в заказе
    его распознает модель."""

    NONE = "None"
    PYTHON = "Python"
    JAVASCRIPT = "JavaScript"
    TYPESCRIPT = "TypeScript"
    CSHARP = "CSharp"
    JAVA = "Java"
    GO = "Go"
    RUST = "Rust"
    CPP = "Cpp"
    SQL = "Sql"
    HTML = "Html"
    OTHER = "Other"


class ScienceField(Indexed):
    """Область науки, если задача научная. Уточняет Domain: область «наука» слишком широка,
    чтобы выучить, кто силен в физике, а кто в экономике."""

    NONE = "None"
    MATHEMATICS = "Mathematics"
    PHYSICS = "Physics"
    CHEMISTRY = "Chemistry"
    BIOLOGY = "Biology"
    MEDICINE = "Medicine"
    ECONOMICS = "Economics"
    COMPUTER_SCIENCE = "ComputerScience"
    SOCIAL_SCIENCES = "SocialSciences"
    HUMANITIES = "Humanities"
    OTHER = "Other"


class TaskKind(Indexed):
    """Тип задачи: что заказчик хочет получить на выходе. NONE означает «не задан» и в векторе
    не светится. Виды сгруппированы так, чтобы каждая группа ложилась на категорию арены или индекс
    Artificial Analysis: начальный прогноз у вида есть без собственных замеров, а различие внутри
    категории выучит журнал."""

    NONE = "None"
    # Письма и коммуникации
    BUSINESS_LETTER = "BusinessLetter"
    COMMERCIAL_OFFER = "CommercialOffer"
    CUSTOMER_REPLY = "CustomerReply"
    INTERNAL_MEMO = "InternalMemo"
    EMAIL_CAMPAIGN = "EmailCampaign"
    # Отчеты и аналитика
    ANALYTICAL_REPORT = "AnalyticalReport"
    FINANCIAL_REPORT = "FinancialReport"
    MARKET_RESEARCH = "MarketResearch"
    DATA_ANALYSIS = "DataAnalysis"
    SUMMARY = "Summary"
    MEETING_MINUTES = "MeetingMinutes"
    # Маркетинг и продажи
    MARKETING_ARTICLE = "MarketingArticle"
    SOCIAL_POST = "SocialPost"
    AD_COPY = "AdCopy"
    PRODUCT_DESCRIPTION = "ProductDescription"
    LANDING_PAGE = "LandingPage"
    PRESS_RELEASE = "PressRelease"
    SALES_SCRIPT = "SalesScript"
    # Документы и право
    LEGAL_DOCUMENT = "LegalDocument"
    LEGAL_ANALYSIS = "LegalAnalysis"
    POLICY = "Policy"
    JOB_DESCRIPTION = "JobDescription"
    # Код и ИТ
    CODE_WRITING = "CodeWriting"
    CODE_REVIEW = "CodeReview"
    TECHNICAL_DOC = "TechnicalDoc"
    # Наука и обучение
    EXPLANATION = "Explanation"
    RESEARCH_REVIEW = "ResearchReview"
    LEARNING_MATERIAL = "LearningMaterial"
    MATH_SOLUTION = "MathSolution"
    # Творчество и медиа
    STORY = "Story"
    SCRIPT = "Script"
    # Консультации и планы
    ADVICE = "Advice"
    BUSINESS_PLAN = "BusinessPlan"
    OTHER = "Other"


class FeedbackType(IntEnum):
    """Тип отзыва. Человеческий стоит дороже, и автоматический оценщик учится по нему."""

    HUMAN = 1
    AUTO = 2


class Capability(IntFlag):
    """Что кандидат умеет. Проверяется до сравнения оценок: кандидат, который заведомо не
    справится, не должен побеждать по цене и скорости."""

    NONE = 0
    CODE = 1
    FORMULAS = 2
    VISION = 4
    TOOLS = 8
    WEB_SEARCH = 16
    ALL = CODE | FORMULAS | VISION | TOOLS | WEB_SEARCH
