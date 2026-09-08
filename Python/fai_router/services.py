from __future__ import annotations

from fai_router import text_metrics
from fai_router.llm.client import OpenRouterClient
from fai_router.llm.spec_input import SpecInputRecognizer
from fai_router.llm.style_classifier import StyleClassifier
from fai_router.specifications import Specifications
from fai_router.tracking import InputFeatures


class InputFeaturesService:
    """Признаки запроса."""

    # Символов на токен
    EST_SYMBOL_PER_TOKEN = 3.0

    _recognizer = SpecInputRecognizer()

    @classmethod
    def get_features(cls, text: str) -> InputFeatures:
        """Признаки объема без обращения к модели."""
        return InputFeatures(
            input_len=len(text) / cls.EST_SYMBOL_PER_TOKEN,
            len_answer=2 * len(text) / cls.EST_SYMBOL_PER_TOKEN,
        )

    @classmethod
    def get_features_full(cls, text: str) -> InputFeatures:
        """Полные признаки: объем оценивается арифметикой, задание распознает модель."""
        features = cls.get_features(text)
        features.input_specifications = cls._recognizer.get_specifications(text)
        return features


class SpecInputService:
    """Получение спецификации на входе, через модель."""

    def __init__(self, llm: OpenRouterClient | None = None):
        self._recognizer = SpecInputRecognizer(llm)

    def get_specifications(self, prompt: str) -> Specifications:
        return self._recognizer.get_specifications(prompt)


class SpecOutputService:
    """Получение спецификации на выходе: структура считается по тексту, стиль распознает
    модель."""

    def __init__(self, llm: OpenRouterClient | None = None):
        self._classifier = StyleClassifier(llm)

    def get_specifications(self, answer: str) -> Specifications:
        if not answer or not answer.strip():
            raise ValueError("Ответ не может быть пустым.")
        spec = text_metrics.measure(answer)
        assessment = self._classifier.assess(answer)
        spec.style_type = assessment.style_type
        spec.term_density = assessment.term_density
        spec.formality_score = assessment.formality_score
        return spec
