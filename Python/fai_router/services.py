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
        """Признаки без обращения к модели: объем по длине текста, задание как у типовой задачи.

        Задание без распознавания неизвестно, а пустая спецификация это не «неизвестно», а крайняя
        точка: стиль «нет», читаемость и формальность ноль. Прогноз качества проецировался на это
        случайное направление, и порядок кандидатов не совпадал даже с их общей силой. Замер «выбрось
        серию рейтингов и предскажи ее» (docs/research/prior-holdout.md): лидер угадан в 3,7 % серий
        против 27,2 % со спецификацией типовой задачи и 34,6 % с полным распознаванием."""
        # Импорт здесь: приор рейтингов сам строится на признаках задачи
        from fai_router.training import benchmark_prior

        return InputFeatures(
            input_len=len(text) / cls.EST_SYMBOL_PER_TOKEN,
            len_answer=2 * len(text) / cls.EST_SYMBOL_PER_TOKEN,
            input_specifications=benchmark_prior.typical_task().input_specifications,
        )

    @classmethod
    def get_features_full(cls, text: str) -> InputFeatures:
        """Полные признаки: объем оценивается арифметикой, задание распознает модель."""
        features = cls.get_features(text)
        features.input_specifications = cls._recognizer.get_specifications(text)
        # Объем ответа берется из распознанного заказа; догадка по длине промпта остается на
        # случай, когда заказ объема не назвал. Раньше цена и время считались только по промпту:
        # «напиши обзор на двадцать тысяч знаков» это короткий запрос, и ход выглядел дешевым и
        # быстрым у всех кандидатов разом
        if features.input_specifications.symbol_length > 0:
            features.len_answer = features.input_specifications.symbol_length / cls.EST_SYMBOL_PER_TOKEN
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
        spec.domain = assessment.domain
        spec.science_field = assessment.science_field
        spec.task_kind = assessment.task_kind
        return spec
