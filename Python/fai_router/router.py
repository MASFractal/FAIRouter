"""Фасад: один объект на весь контур. Выбор исполнителя, выполнение с запасным вариантом, замер
ответа, оценка судьи, запись в журнал, отзывы и обучение."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from fai_router import env
from fai_router.diff_spec import DiffSpec
from fai_router.enums import Capability, FeedbackType
from fai_router.judge import Judge
from fai_router.llm.client import OpenRouterClient
from fai_router.persistence import SqliteTraceStore, SqliteWeightsStore
from fai_router.routed_element import RoutedElement
from fai_router.services import SpecOutputService
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert
from fai_router.training import JudgeTrainer, RouterTrainer

Messages = list[dict[str, str]]


@dataclass
class Completion:
    """Ответ исполнителя вместе с расходом токенов. Исполнитель вправе вернуть и просто строку."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


# Исполнитель: как получить ответ выбранного кандидата на сообщения диалога
Executor = Callable[[RoutedElement, Messages], "Completion | str"]


@dataclass
class RouterAnswer:
    """Итог одного хода."""

    text: str
    winner: str
    trace: Tracert
    requested: Specifications | None
    actual: Specifications | None = None
    score: float | None = None
    critic: DiffSpec | None = None
    round_id: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class FaiRouter:
    """Роутер целиком. Вызывающему остается назвать кандидатов и сказать, как их спрашивать."""

    def __init__(
        self,
        candidates: Iterable[RoutedElement],
        execute: Executor,
        database_path: str | None = None,
        topk: int = 5,
        measure: bool = True,
        llm: OpenRouterClient | None = None,
    ):
        self.candidates = list(candidates)
        self._execute = execute
        self.topk = topk
        # Замер ответа стоит одного обращения к модели на ход; его можно отключить
        self.measure = measure
        self.judge = Judge()
        self.traces = SqliteTraceStore(database_path) if database_path else None
        self.weights = SqliteWeightsStore(database_path) if database_path else None
        self._measurer = SpecOutputService()
        self._router_trainer = RouterTrainer(learning_rate=0.05)
        self._judge_trainer = JudgeTrainer(self.judge, learning_rate=0.5)
        if llm is not None:
            Settings.llm = llm
        if self.weights is not None:
            self.load()

    @classmethod
    def from_openrouter(
        cls,
        api_key: str,
        model_ids: Iterable[str],
        database_path: str | None = None,
        judge_model: str = "openai/gpt-4o-mini",
        tokens_per_second: dict[str, float] | None = None,
        **kwargs,
    ) -> "FaiRouter":
        """Роутер над моделями OpenRouter: цены и возможности из каталога, ответы через него же.
        Скорость поставщик не публикует, ее можно передать словарем по идентификаторам."""
        from fai_router import catalog

        known = {model.id: model for model in catalog.fetch()}
        speeds = tokens_per_second or {}
        candidates = []
        for model_id in model_ids:
            if model_id not in known:
                raise ValueError(f"Модели {model_id} нет в каталоге OpenRouter.")
            candidates.append(catalog.create_element(known[model_id], speeds.get(model_id, 50.0)))

        clients: dict[str, OpenRouterClient] = {}

        def execute(candidate: RoutedElement, messages: Messages) -> Completion:
            client = clients.setdefault(candidate.name, OpenRouterClient(api_key, candidate.name))
            data = client.complete_full(messages)
            prompt_tokens, completion_tokens = OpenRouterClient.usage(data)
            return Completion(data["choices"][0]["message"]["content"] or "", prompt_tokens, completion_tokens)

        return cls(candidates, execute, database_path, llm=OpenRouterClient(api_key, judge_model), **kwargs)

    def ask(self, prompt: str, required: Capability = Capability.NONE) -> RouterAnswer:
        """Один ход по тексту запроса."""
        return self.ask_messages([{"role": "user", "content": prompt}], required)

    def ask_messages(self, messages: Messages, required: Capability = Capability.NONE) -> RouterAnswer:
        """Один ход по диалогу: задача распознается по последнему сообщению пользователя, а
        исполнителю уходит весь диалог целиком."""
        prompt = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if not prompt.strip():
            raise ValueError("В диалоге нет сообщения пользователя, задачу распознать не из чего.")

        trace = env.route(prompt, self.candidates, self.topk, required)
        completion = env.execute(trace, lambda candidate: self._execute(candidate, messages))
        if isinstance(completion, str):
            completion = Completion(completion)

        answer = RouterAnswer(completion.text, trace.winner.name, trace, trace.requested_spec,
                              prompt_tokens=completion.prompt_tokens,
                              completion_tokens=completion.completion_tokens)

        if self.measure and completion.text.strip():
            answer.actual = self._measurer.get_specifications(completion.text)
            answer.score = self.judge.rate(trace, trace.requested_spec, answer.actual)
            answer.critic = Judge.criticize(trace.requested_spec, answer.actual)

        if self.traces is not None:
            answer.round_id = self.traces.append(trace, trace.requested_spec, answer.actual, prompt)
            # Отзыв критика ставится сразу; человеческий, если придет, его перезапишет
            if answer.critic is not None:
                self.traces.set_feedback(answer.round_id,
                                         Feedback(FeedbackType.AUTO, 1 - answer.critic.total_deviation))
        return answer

    def feedback(self, round_id: int, score: float, human: bool = True) -> None:
        """Отзыв на ход: единица означает «нравится», ноль означает «нет»."""
        self._require_journal().set_feedback(
            round_id, Feedback(FeedbackType.HUMAN if human else FeedbackType.AUTO, score))

    def train(self, epochs: int = 1) -> float:
        """Обучение по накопленному журналу. Возвращает ошибку последней эпохи."""
        journal = self._require_journal()
        # Среднее по задачам задается один раз: веса обучены в пространстве с этим средним
        if Settings.task_mean is None:
            Settings.task_mean = journal.get_feature_mean()
        sample = journal.read_rated(self.candidates)
        loss = 0.0
        for _ in range(epochs):
            loss = 0.0
            for training_round in sample:
                loss += self._router_trainer.train(training_round.trace, training_round.feedback)
                # Судья учится только у человека. Автоотзыв это разбор расхождений по пунктам, и
                # учить по нему судью значило бы подгонять одну автоматическую оценку под другую,
                # а человек из этого круга выпадал бы совсем
                if (training_round.feedback.ftype == FeedbackType.HUMAN
                        and training_round.requested is not None and training_round.actual is not None):
                    loss += self._judge_trainer.train(training_round.requested, training_round.actual,
                                                      training_round.feedback.score)
        journal.load_statistics(self.candidates)
        return loss

    def save(self) -> None:
        store = self._require_weights()
        store.save_elements(self.candidates)
        store.save_judge(self.judge)
        if Settings.task_mean is not None:
            store.save_task_mean(Settings.task_mean)

    def load(self) -> None:
        store = self._require_weights()
        store.load_elements(self.candidates)
        store.load_judge(self.judge)
        mean = store.load_task_mean()
        if mean is not None:
            Settings.task_mean = mean
        if self.traces is not None:
            self.traces.load_statistics(self.candidates)

    def _require_journal(self) -> SqliteTraceStore:
        if self.traces is None:
            raise RuntimeError("Журнал не задан: создайте роутер с database_path.")
        return self.traces

    def _require_weights(self) -> SqliteWeightsStore:
        if self.weights is None:
            raise RuntimeError("Хранилище не задано: создайте роутер с database_path.")
        return self.weights
