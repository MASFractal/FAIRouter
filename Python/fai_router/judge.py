from __future__ import annotations

import numpy as np

from fai_router.content_review import ContentReview
from fai_router.diff_spec import DiffSpec
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import Tracert


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


class Judge:
    """Судья, автоматическая оценка качества решения. Замером факта судья не занимается:
    обе спецификации приходят к нему готовыми."""

    def __init__(self):
        # Матрица трансформации (обучаемая). Единичная на старте означает наивное допущение,
        # будто факт обязан совпасть с заказом
        self.transformer_w: np.ndarray = np.eye(Settings.features_spec_dim())

    def get_score(self, input_spec: Specifications, actual_spec: Specifications) -> float:
        """Близость факта к заказу, пропущенному через обучаемую матрицу."""
        transformed = self.transformer_w @ input_spec.feature_vector()
        return cosine(actual_spec.feature_vector(), transformed)

    def rate(self, trace: Tracert, input_spec: Specifications, actual_spec: Specifications) -> float:
        """Проставляет оценку хода в его трассировку и тем замыкает круг: ход состоялся,
        судья его оценил, и трассировка стала обучающим примером."""
        trace.score = self.get_score(input_spec, actual_spec)
        return trace.score

    @staticmethod
    def criticize(input_spec: Specifications, actual_spec: Specifications,
                  content: ContentReview | None = None) -> DiffSpec:
        """Режим критика: расхождения между заданием и фактом по каждому пункту; с оценкой
        содержания и по содержанию."""
        return DiffSpec.compare(input_spec, actual_spec, content)

    @staticmethod
    def assess(critic: DiffSpec, content: ContentReview | None) -> float:
        """Итоговая оценка ответа: содержание и форма с долями из Settings.content_weight. Форма
        здесь это доля выполненных пунктов формы критика: число объяснимое и не зависящее от
        обучаемой матрицы. Без оценки содержания итог равен форме."""
        form_score = 1.0 - critic.form_deviation
        if content is None:
            return form_score
        return Settings.content_weight * content.score + (1.0 - Settings.content_weight) * form_score

    @staticmethod
    def report(critic: DiffSpec, content: ContentReview | None) -> str:
        """Отчет: оценки содержания, формы и итог, затем все проваленные пункты задания и
        замечания судьи."""
        form = f"{1.0 - critic.form_deviation:.2f}"
        head = (f"Форма {form}" if content is None
                else f"Содержание {content.score:.2f}, форма {form}, итог {Judge.assess(critic, content):.2f}")
        lines = [head]
        lines += [f"{item.field}: заказано {item.requested}, получено {item.actual}" for item in critic.mismatches]
        lines += [f"- {issue}" for issue in (content.issues if content is not None else [])]
        return "\n".join(lines)
