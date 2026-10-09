from __future__ import annotations

import logging
from typing import Protocol

from fai_router import text_metrics
from fai_router.llm.spec_input import SpecInputRecognizer
from fai_router.llm.style_classifier import StyleClassifier
from fai_router.specifications import Specifications
from fai_router.tracking import InputFeatures

log = logging.getLogger("fai_router")


class SpecSource(Protocol):
    """Кто дает спецификацию текста: заказа по запросу или факта по ответу."""

    def get_specifications(self, text: str) -> Specifications: ...


class InputFeaturesService:
    """Признаки запроса."""

    # Символов на токен
    EST_SYMBOL_PER_TOKEN = 3.0

    # Распознаватель по умолчанию: клиент модели берет общий Settings.llm в момент вызова
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
    def get_features_full(cls, text: str, specs: SpecSource | None = None) -> InputFeatures:
        """Полные признаки: объем оценивается арифметикой, задание распознает модель. Сбой
        распознавания хода не роняет: признаки остаются как у get_features."""
        features = cls.get_features(text)
        recognized = cls.recognize(text, specs)
        if recognized is not None:
            cls.apply(features, recognized)
        return features

    @classmethod
    def recognize(cls, text: str, specs: SpecSource | None = None) -> Specifications | None:
        """Распознанное задание; None, если распознать не удалось. Ловится любая ошибка, кроме
        прерывания: обрезанный или негодный ответ модели, сеть, отказ поставщика, исчерпанный бюджет.
        Прежде такой сбой ронял ход до выбора исполнителя. В журнал пишется только тип ошибки: ее
        текст может нести начало запроса пользователя."""
        try:
            return (specs or cls._recognizer).get_specifications(text)
        except Exception as error:  # noqa: BLE001 - сбой распознавания заменяется типовой задачей
            log.warning("Задание не распознано (%s), выбор идет по типовой задаче.", type(error).__name__)
            return None

    @classmethod
    def apply(cls, features: InputFeatures, recognized: Specifications) -> None:
        """Ставит распознанное задание в признаки. Объем ответа берется из заказа; догадка по длине
        промпта остается на случай, когда заказ объема не назвал. Раньше цена и время считались только
        по промпту: «напиши обзор на двадцать тысяч знаков» это короткий запрос, и ход выглядел дешевым
        и быстрым у всех кандидатов разом."""
        features.input_specifications = recognized
        if recognized.symbol_length > 0:
            features.len_answer = recognized.symbol_length / cls.EST_SYMBOL_PER_TOKEN


class SpecInputService:
    """Получение спецификации на входе, через модель."""

    def __init__(self, llm=None):
        self._recognizer = SpecInputRecognizer(llm)

    def get_specifications(self, prompt: str) -> Specifications:
        return self._recognizer.get_specifications(prompt)


class SpecOutputService:
    """Получение спецификации на выходе: структура считается по тексту, стиль распознает
    модель."""

    def __init__(self, llm=None):
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
