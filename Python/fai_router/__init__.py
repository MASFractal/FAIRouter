"""FAI.Router на Python: выбор исполнителя, судья результата и обучение на отзывах."""

from fai_router.enums import Capability, FeedbackType, Style
from fai_router.settings import RouteWeights, Settings
from fai_router.specifications import Specifications
from fai_router.diff_spec import DiffSpec, SpecDeviation
from fai_router.judge import Judge
from fai_router.routed_element import RoutedElement
from fai_router.tracking import Feedback, InputFeatures, Tracert
from fai_router import env
from fai_router.router import Completion, FaiRouter, RouterAnswer

__all__ = [
    "Capability", "FeedbackType", "Style", "RouteWeights", "Settings", "Specifications",
    "DiffSpec", "SpecDeviation", "Judge", "RoutedElement",
    "Feedback", "InputFeatures", "Tracert", "env",
    "FaiRouter", "RouterAnswer", "Completion",
]
