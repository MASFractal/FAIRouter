"""FAI.Router на Python: выбор исполнителя, судья результата и обучение на отзывах."""

from fai_router.enums import Capability, Domain, FeedbackType, ProgrammingLanguage, ScienceField, Style, TaskKind
from fai_router.settings import RouteWeights, Settings, SufficiencyBar
from fai_router.specifications import Specifications
from fai_router.diff_spec import DiffSpec, SpecDeviation
from fai_router.content_review import ConstraintCheck, ContentCriterion, ContentReview, FactClaim, PointCoverage
from fai_router.judge import Judge
from fai_router.routed_element import RoutedElement
from fai_router.tracking import Feedback, InputFeatures, Tracert
from fai_router import env
from fai_router.router import Completion, FaiRouter, RouterAnswer

__all__ = [
    "Capability", "FeedbackType", "Style", "Domain", "ProgrammingLanguage", "ScienceField", "TaskKind",
    "RouteWeights", "Settings", "SufficiencyBar", "Specifications",
    "DiffSpec", "SpecDeviation", "ContentReview", "ContentCriterion", "FactClaim", "PointCoverage", "ConstraintCheck", "Judge", "RoutedElement",
    "Feedback", "InputFeatures", "Tracert", "env",
    "FaiRouter", "RouterAnswer", "Completion",
]
