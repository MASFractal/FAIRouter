from __future__ import annotations

import numpy as np

from fai_router.enums import FeedbackType
from fai_router.settings import Settings
from fai_router.tracking import Feedback, Tracert


class RouterTrainer:
    """Контрастивное обучение роутера: вектор победителя сдвигается так, чтобы на похожих задачах
    впереди оказывался тот, чей ответ понравился. Учится порядок, то есть какой кандидат уместнее
    именно здесь. Правило общее с версией на C# (RouterTrainer).

    Градиент идет только в победителя. Соперников на этом ходе не пробовали, и отзыв о них ничего
    не говорит: прежде их векторы сдвигались по чужому отзыву, и модель, которую ни разу не звали,
    проседала от чужих лайков. Их оценки в штрафе неподвижны.

    Штраф усредняется по соперникам, а не суммируется: иначе ход с пятью соперниками двигал
    победителя впятеро сильнее хода с одним.

    Состояния между шагами тренер не держит: градиент считается от нынешних векторов кандидатов,
    а вектор победителя подменяется целиком, и соседний выбор видит старый вектор или новый."""

    # Требуемый отрыв победителя: без зазора обучение останавливается, едва порядок стал верным
    MARGIN = 0.1
    # Оценка, начиная с которой отзыв считается положительным
    LIKE_THRESHOLD = 0.5
    # Во сколько раз автоотзыв слабее человеческого; общее значение живет в Settings
    AUTO_FEEDBACK_WEIGHT = Settings.AUTO_FEEDBACK_WEIGHT

    def __init__(self, learning_rate: float = 0.01, exploration_weight: float = 1.0):
        """exploration_weight: множитель шага на ходах, отданных не лидеру (is_exploration).
        Единица: разведочный ход учит так же, как обычный; больше единицы поднимает голос ходов, где
        выбор не совпал с мнением роутера, то есть компенсирует смещение выборки к лидерам."""
        self._lr = learning_rate
        self._exploration_weight = exploration_weight

    def train(self, trace: Tracert, feedback: Feedback) -> float:
        """Один шаг по трассировке хода. Возвращает ошибку до шага."""
        winner = trace.winner
        rivals = [element for element in trace.top_k_elements if element is not winner]
        # Соперников нет: контрастивной паре не из чего взяться
        if not rivals:
            return 0.0

        # Тот же вид признаков, что и при подсчете прогноза: иначе вектор учился бы в одном
        # пространстве, а работал в другом
        mean = Settings.task_mean if winner.task_mean is None else winner.task_mean
        task = Settings.center(trace.input_feature_vector, mean)
        vector = winner.ideal_match_vector
        winner_score = float(task @ vector)
        liked = feedback.score >= self.LIKE_THRESHOLD

        # Сила отзыва, а не только его знак. Без множителя посредственный, но одобренный ответ
        # двигал бы веса так же, как отличный, и разведка теряла бы смысл
        weight = abs(feedback.score - self.LIKE_THRESHOLD) / self.LIKE_THRESHOLD
        if feedback.ftype != FeedbackType.HUMAN:
            weight *= Settings.AUTO_FEEDBACK_WEIGHT
        if trace.is_exploration:
            weight *= self._exploration_weight

        loss, active = 0.0, 0
        for rival in rivals:
            rival_score = rival.get_quality_score(trace.input_feature_vector)
            # Если понравилось, победитель должен быть выше соперника, если нет, то ниже
            gap = winner_score - rival_score if liked else rival_score - winner_score
            penalty = self.MARGIN - gap
            if not np.isfinite(penalty) or penalty <= 0:
                continue
            loss += penalty
            active += 1

        loss *= weight / len(rivals)
        if active == 0 or weight <= 0:
            return float(loss)

        # Производная среднего штрафа по вектору победителя: задача со знаком на долю активных пар
        step = self._lr * weight * active / len(rivals) * (1 if liked else -1)
        trained = vector + task * step
        if np.isfinite(trained).all():
            winner.ideal_match_vector = trained
        return float(loss)
