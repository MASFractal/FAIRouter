"""Фасад: один объект на весь контур. Выбор исполнителя, выполнение с запасным вариантом, замер
ответа, оценка судьи, запись в журнал, отзывы и обучение."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Iterable

from fai_router import env
from fai_router.content_review import ContentReview
from fai_router.diff_spec import DiffSpec
from fai_router.enums import Capability, FeedbackType
from fai_router import catalog as catalog_module
from fai_router.judge import Judge
from fai_router.llm.client import FRACTALROUTER_URL, OPENROUTER_URL, OpenRouterClient
from fai_router.llm.content_judge import ContentJudge
from fai_router.persistence import SqliteTraceStore, SqliteWeightsStore
from fai_router.routed_element import RoutedElement
from fai_router.services import SpecOutputService
from fai_router.settings import RouteWeights, Settings, SufficiencyBar
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert
from fai_router.training import Calibration, JudgeTrainer, RouterTrainer

if TYPE_CHECKING:
    from fai_router.benchmarks import BenchmarkSnapshot

log = logging.getLogger("fai_router")

Messages = list[dict[str, str]]

# Цены кандидата за миллион токенов в валюте поставщика: (вход, выход)
Prices = dict[str, tuple[float, float]]


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
    # Оценка содержания и итоговая оценка по содержанию и форме; None, если замер отключен
    content: ContentReview | None = None
    assessment: float | None = None
    # Планка достаточности: истина, если кто-то до нее дотянул, ложь, если ход отдан сильнейшему
    # при недоборе и человека стоит предупредить; None, если выбор шел без планки
    reached: bool | None = None


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
        content_judge: ContentJudge | None = None,
        weights: "RouteWeights | str | None" = None,
        bar: "float | SufficiencyBar | None" = None,
        min_ratings: int = 3,
        temperature_scale: float | None = None,
    ):
        self.candidates = list(candidates)
        # Свой множитель температуры роутера; None означает общий из Settings. Фабрики ставят
        # сюда пониженный Settings.PRIOR_TEMPERATURE_SCALE, когда все кандидаты стартуют с
        # начальных весов по рейтингам. Веса, заданные готовым RouteWeights, множитель не трогает
        self.temperature_scale = temperature_scale
        self._explicit_weights = isinstance(weights, RouteWeights)
        # Планка достаточности на все ходы: число от 0 до 1 это обязательная вероятность лайка,
        # калибровка к ней подбирается по журналу человеческих отзывов; готовая SufficiencyBar
        # берется как есть. None означает выбор по метрике R без планки
        self.bar = bar
        # Сколько человеческих отзывов нужно, чтобы калибровка планки по журналу считалась
        # осмысленной; до этого ход идет без планки
        self.min_ratings = min_ratings
        # Профиль весов на все ходы: quality, balance, price либо свои RouteWeights; None означает
        # веса из Settings. Ход может назвать свой профиль и перекрыть этот
        self.route_weights = RouteWeights.profile(weights)
        self._execute = execute
        self.topk = topk
        # Оценка ответа по форме и содержанию стоит двух обращений к модели на ход; ее можно отключить
        self.measure = measure
        # Свой судья содержания нужен, например, для проверки фактов по вебу
        self.content_judge = content_judge or ContentJudge()
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
    def from_fractalrouter(
        cls,
        api_key: str,
        model_ids: "str | Iterable[str]" = catalog_module.POPULAR,
        database_path: str | None = None,
        prices: Prices | None = None,
        judge_model: str = "openai/gpt-4o-mini",
        base_url: str = FRACTALROUTER_URL,
        **kwargs,
    ) -> "FaiRouter":
        """Роутер над моделями FractalRouter (fractalrouter.ru): цены и возможности из его
        каталога, ответы кандидатов и работа судьи через него же, одним ключом. Модели: список
        идентификаторов, «popular» (популярные из комплекта, по умолчанию) или «all» (весь каталог).
        Цены в рублях за миллион токенов; модели, которой в каталоге нет, цену задает словарь
        prices. Остальные доводы те же, что у from_openai_compatible."""
        from fai_router import catalog

        return cls.from_openai_compatible(
            base_url, api_key, model_ids, database_path=database_path, prices=prices,
            judge_model=judge_model, catalog_fetch=lambda: catalog.fetch_fractalrouter(api_key), **kwargs)

    @classmethod
    def from_openrouter(
        cls,
        api_key: str,
        model_ids: "str | Iterable[str]" = catalog_module.POPULAR,
        database_path: str | None = None,
        judge_model: str = "openai/gpt-4o-mini",
        tokens_per_second: dict[str, float] | None = None,
        **kwargs,
    ) -> "FaiRouter":
        """Роутер над моделями OpenRouter: цены и возможности из каталога, ответы через него же.
        Скорость поставщик не публикует, ее можно передать словарем по идентификаторам."""
        return cls.from_openai_compatible(OPENROUTER_URL, api_key, model_ids, database_path=database_path,
                                          judge_model=judge_model, tokens_per_second=tokens_per_second,
                                          **kwargs)

    @classmethod
    def from_openai_compatible(
        cls,
        base_url: str,
        api_key: str,
        model_ids: "str | Iterable[str]" = catalog_module.POPULAR,
        database_path: str | None = None,
        prices: Prices | None = None,
        judge_model: str = "openai/gpt-4o-mini",
        tokens_per_second: dict[str, float] | None = None,
        benchmarks: "BenchmarkSnapshot | str | None" = None,
        timeout: float = 120.0,
        catalog_fetch: "Callable[[], Iterable[catalog.ModelInfo]] | None" = None,
        **kwargs,
    ) -> "FaiRouter":
        """Роутер над любым поставщиком по протоколу OpenAI chat completions: base_url это адрес
        вида https://host/v1, ключ уходит заголовком Bearer. Через того же поставщика и тем же
        ключом работает модель-судья judge_model.

        Модели задаются списком идентификаторов либо именем набора: «popular» это популярные из
        комплекта (data/popular_models.json), которые есть в каталоге поставщика, «all» это весь его
        каталог. Профиль весов weights (quality, balance, price) действует на все ходы, как и
        планка достаточности bar: обязательная вероятность лайка от 0 до 1, калибруется по
        журналу человеческих отзывов.

        Цены кандидатов берутся из prices, (вход, выход) за миллион токенов в валюте поставщика;
        модели, которых там нет, ищутся в каталоге catalog_fetch (по умолчанию каталог OpenRouter)
        по идентификатору. Модель без цены ни там, ни там это ошибка: без цены роутеру нечего
        взвешивать.

        Начальные веса кандидатов берутся из снимка внешних замеров (benchmarks: снимок, путь к
        нему или None для снимка из комплекта), поэтому роутер небесполезен с первого хода, а
        обучение на отзывах его уточняет. Пустая база на старте в порядке: веса в нее попадают
        при save(), ходы и отзывы при ask() и feedback()."""
        from fai_router import benchmarks as benchmarks_module
        from fai_router import catalog

        snapshot = benchmarks_module.resolve(benchmarks)
        if snapshot is None:
            log.warning("Снимок замеров не найден: кандидаты стартуют со случайных весов, "
                        "пока не накопятся отзывы.")
        prices = prices or {}
        speeds = tokens_per_second or {}
        known: dict[str, catalog.ModelInfo] = {}
        if isinstance(model_ids, str) or any(model_id not in prices for model_id in model_ids):
            known = {model.id: model for model in (catalog_fetch or catalog.fetch)()}
        model_ids, strict = catalog.select(model_ids, known)
        if not model_ids:
            raise ValueError("Список моделей пуст: роутеру не из кого выбирать.")
        candidates = []
        for model_id in model_ids:
            if model_id in prices:
                inp, outp = prices[model_id]
                info = catalog.ModelInfo(model_id, model_id, float(inp), float(outp), 0, 0,
                                         Capability.CODE | Capability.FORMULAS)
            elif model_id in known:
                info = known[model_id]
            else:
                raise ValueError(
                    f"У модели {model_id} нет цены в prices, и в каталоге поставщика ее нет. "
                    f"Задайте цену: prices={{\"{model_id}\": (вход, выход)}} за миллион токенов.")
            candidates.append(catalog.create_element(info, speeds.get(model_id), snapshot))

        clients: dict[str, OpenRouterClient] = {}

        def execute(candidate: RoutedElement, messages: Messages) -> Completion:
            client = clients.setdefault(
                candidate.name, OpenRouterClient(api_key, candidate.name, timeout=timeout, base_url=base_url))
            data = client.complete_full(messages)
            prompt_tokens, completion_tokens = OpenRouterClient.usage(data)
            return Completion(data["choices"][0]["message"]["content"] or "", prompt_tokens, completion_tokens)

        # Все кандидаты стартуют с начальных весов по рейтингам: долгая разведка не нужна, и
        # множитель температуры берется пониженный. Хотя бы один без рейтингов, тогда общий
        from fai_router.training import benchmark_prior

        informed = snapshot is not None and all(
            benchmark_prior.vector(snapshot, model_id) is not None for model_id in model_ids)
        if informed:
            kwargs.setdefault("temperature_scale", Settings.PRIOR_TEMPERATURE_SCALE)

        judge = OpenRouterClient(api_key, judge_model, timeout=timeout, base_url=base_url)
        return cls(candidates, execute, database_path, llm=judge, **kwargs)

    def ask(self, prompt: str, required: Capability = Capability.NONE,
            weights: "RouteWeights | str | None" = None,
            bar: "float | SufficiencyBar | None" = None) -> RouterAnswer:
        """Один ход по тексту запроса. Профиль весов на этот ход: quality, balance, price либо
        свои RouteWeights; не задан, тогда профиль роутера. Планка на этот ход: обязательная
        вероятность лайка либо готовая SufficiencyBar; не задана, тогда планка роутера."""
        return self.ask_messages([{"role": "user", "content": prompt}], required, weights, bar)

    def ask_messages(self, messages: Messages, required: Capability = Capability.NONE,
                     weights: "RouteWeights | str | None" = None,
                     bar: "float | SufficiencyBar | None" = None) -> RouterAnswer:
        """Один ход по диалогу: задача распознается по последнему сообщению пользователя, а
        исполнителю уходит весь диалог целиком."""
        prompt = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if not prompt.strip():
            raise ValueError("В диалоге нет сообщения пользователя, задачу распознать не из чего.")

        turns = sum(1 for m in messages if m.get("role") == "user")
        trace = env.route(prompt, self.candidates, self.topk, required,
                          weights=self._weights_for(weights), turns=turns,
                          bar=self.sufficiency_bar(self.bar if bar is None else bar))
        completion = env.execute(trace, lambda candidate: self._execute(candidate, messages))
        if isinstance(completion, str):
            completion = Completion(completion)

        answer = RouterAnswer(completion.text, trace.winner.name, trace, trace.requested_spec,
                              prompt_tokens=completion.prompt_tokens,
                              completion_tokens=completion.completion_tokens,
                              reached=trace.bar_reached)

        if self.measure and completion.text.strip():
            self._measure(answer, prompt)

        if self.traces is not None:
            answer.round_id = self.traces.append(trace, trace.requested_spec, answer.actual, prompt)
            # Автоотзыв это итоговая оценка по содержанию и форме; человеческий его перезапишет
            if answer.assessment is not None:
                self.traces.set_feedback(answer.round_id, Feedback(FeedbackType.AUTO, answer.assessment))
        return answer

    def _weights_for(self, weights: "RouteWeights | str | None") -> RouteWeights | None:
        """Веса хода: названные для хода, иначе веса роутера, иначе общие из Settings. Свой
        множитель температуры роутера подставляется в веса по умолчанию и в именованные профили;
        веса, переданные готовым RouteWeights, берутся как есть."""
        if isinstance(weights, RouteWeights):
            return weights
        named = RouteWeights.profile(weights)
        if named is None and self._explicit_weights:
            return self.route_weights
        chosen = named or self.route_weights
        if self.temperature_scale is None:
            return chosen
        return replace(chosen or Settings.current(), temperature_scale=self.temperature_scale)

    def sufficiency_bar(self, bar: "float | SufficiencyBar | None") -> SufficiencyBar | None:
        """Планка для хода. Готовая SufficiencyBar берется как есть. Число это уровень, а
        калибровка к нему подбирается по журналу: пары «прогноз качества победителя на той задаче
        и оценка человека», доля лайков как усадка для малоизученных. Пока человеческих отзывов
        меньше min_ratings, планки нет и ход идет по метрике R: калибровать не на чем, а планка
        без калибровки отсекала бы наугад."""
        if bar is None or isinstance(bar, SufficiencyBar):
            return bar
        if not 0.0 <= bar <= 1.0:
            raise ValueError(f"Планка {bar} вне диапазона: нужна вероятность лайка от 0 до 1.")
        if self.traces is None:
            raise RuntimeError("Планка по уровню требует журнала: создайте роутер с database_path "
                               "либо передайте готовую SufficiencyBar с калибровкой.")
        pairs = self.calibration_pairs()
        if len(pairs) < self.min_ratings:
            log.info("Человеческих отзывов %d из %d нужных: ход без планки.", len(pairs), self.min_ratings)
            return None
        rate = sum(score for _, score in pairs) / len(pairs)
        return SufficiencyBar(bar, Calibration.fit(pairs), prior_rate=rate)

    def calibration_pairs(self) -> list[tuple[float, float]]:
        """Пары для калибровки планки из журнала: прогноз качества победителя на признаках той
        задачи при нынешних весах и оценка человека. Автоотзывы не берутся: планка обещает
        вероятность лайка человека, а не согласие судьи с самим собой."""
        return [(training_round.trace.winner.get_quality_score(training_round.trace.input_feature_vector),
                 training_round.feedback.score)
                for training_round in self._require_journal().read_rated(self.candidates)
                if training_round.feedback.ftype == FeedbackType.HUMAN]

    def _measure(self, answer: RouterAnswer, prompt: str) -> None:
        """Замер и оценка ответа. Сбой судьи (сеть, поставщик) ход не роняет: исполнитель уже
        ответил, и этот ответ дороже оценки. Ход без замера идет в журнал без автоотзыва."""
        try:
            answer.actual = self._measurer.get_specifications(answer.text)
            answer.content = self.content_judge.review(prompt, answer.requested, answer.text)
            answer.score = self.judge.rate(answer.trace, answer.requested, answer.actual)
            answer.critic = Judge.criticize(answer.requested, answer.actual, answer.content)
            answer.assessment = Judge.assess(answer.critic, answer.content)
        except Exception as error:  # noqa: BLE001 - любой сбой замера, ответ отдается без оценки
            log.warning("Ответ получен, но замерить его не удалось: %s", error)

    def feedback(self, round_id: int, score: float, human: bool = True) -> None:
        """Отзыв человека на ход: число от 0 до 1. Единица означает отличный ответ, ноль
        никуда не годный, 0,5 так себе. Отзыв перезаписывает автоотзыв судьи, на нем учатся и
        роутер, и судья. Отзывом вне диапазона роутер не кормится."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Отзыв {score} вне диапазона: нужно число от 0 (плохо) до 1 (отлично).")
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
