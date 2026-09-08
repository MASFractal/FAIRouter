from __future__ import annotations

import numpy as np

from fai_router.routed_element import RoutedElement
from fai_router.settings import Settings
from fai_router.tracking import Feedback, Tracert


class RouterTrainer:
    """Контрастивное обучение роутера: векторы победителя и соперников разводятся так, чтобы на
    похожих задачах впереди оказывался тот, чей ответ понравился. Учится порядок, то есть какой
    кандидат уместнее именно здесь."""

    # Требуемый отрыв победителя: без зазора обучение останавливается, едва порядок стал верным
    MARGIN = 0.1
    # Оценка, начиная с которой отзыв считается положительным
    LIKE_THRESHOLD = 0.5

    def __init__(self, learning_rate: float = 0.01):
        self._lr = learning_rate

    def train(self, trace: Tracert, feedback: Feedback) -> float:
        """Один шаг по трассировке хода. Возвращает ошибку до шага."""
        rivals = [element for element in trace.top_k_elements if element is not trace.winner]
        # Соперников нет: контрастивной паре не из чего взяться
        if not rivals:
            return 0.0

        # Тот же вид признаков, что и при подсчете прогноза
        task = Settings.center(trace.input_feature_vector)
        liked = feedback.score >= self.LIKE_THRESHOLD
        # Сила отзыва, а не только его знак. Без множителя посредственный, но одобренный ответ
        # двигал бы веса так же, как отличный, и разведка теряла бы смысл
        weight = abs(feedback.score - self.LIKE_THRESHOLD) / self.LIKE_THRESHOLD

        winner = trace.winner
        gradients: dict[int, np.ndarray] = {}
        loss = 0.0

        winner_score = float(winner.ideal_match_vector @ task)
        for rival in rivals:
            rival_score = float(rival.ideal_match_vector @ task)
            # Если понравилось, победитель должен быть выше соперника, если нет, то ниже
            gap = winner_score - rival_score if liked else rival_score - winner_score
            penalty = max(0.0, self.MARGIN - gap)
            loss += penalty
            if penalty <= 0:
                continue
            # Градиент штрафа relu(margin - gap) по векторам: у того, кто должен быть выше,
            # минус задача, у того, кто ниже, плюс задача
            up, down = (winner, rival) if liked else (rival, winner)
            gradients[id(up)] = gradients.get(id(up), 0) - task
            gradients[id(down)] = gradients.get(id(down), 0) + task

        # Роутер считает по вектору кандидата, поэтому шаг делается прямо в нем
        for element in [winner, *rivals]:
            gradient = gradients.get(id(element))
            if gradient is not None:
                element.ideal_match_vector = element.ideal_match_vector - self._lr * weight * gradient

        return float(weight * loss)
