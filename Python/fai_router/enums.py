from enum import Enum, IntEnum, IntFlag


class Style(Enum):
    """Стиль текста. Порядок значений задает разряды кода «один из многих» в векторе признаков,
    поэтому переставлять их нельзя: сохраненные веса перестанут соответствовать координатам."""

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

    @property
    def index(self) -> int:
        return list(Style).index(self)


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
