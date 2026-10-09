"""Фасад: один объект на весь контур. Выбор исполнителя, выполнение с запасным вариантом, замер
ответа, оценка судьи, запись в журнал, отзывы и обучение. Память (журнал, обучение, веса,
калибровка) живет в RouterMemory, фасад ведет ход. Устроен так же, как FaiRouter в версии на C#."""

from __future__ import annotations

import inspect
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Iterable

from fai_router import catalog as catalog_module
from fai_router import env
from fai_router.content_review import ContentReview
from fai_router.diff_spec import DiffSpec
from fai_router.enums import Capability
from fai_router.judge import Judge
from fai_router.llm.client import FRACTALROUTER_URL, OPENROUTER_URL, OpenRouterClient
from fai_router.llm.content_judge import ContentJudge
from fai_router.memory import CalibrationState, RouterMemory, TrainingLoss
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService, SpecInputService, SpecOutputService
from fai_router.settings import RouteWeights, Settings, SufficiencyBar
from fai_router.specifications import Specifications
from fai_router.tracking import Tracert

if TYPE_CHECKING:
    from fai_router.benchmarks import BenchmarkSnapshot
    from fai_router.persistence import SqliteTraceStore, SqliteWeightsStore

log = logging.getLogger("fai_router")

# Сообщение диалога в формате OpenAI: content строкой либо массивом частей
Messages = list[dict[str, Any]]

# Цены кандидата за миллион токенов в валюте поставщика: (вход, выход)
Prices = dict[str, tuple[float, float]]

# Модель-судья по умолчанию: распознает задание и разбирает ответ
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

# Снимок старше этого числа дней описывает прошлое поколение моделей
SNAPSHOT_MAX_AGE_DAYS = 180

# Потолок ответа исполнителя по заказанному объему: вдвое больше заказа в токенах и не меньше
# MIN_ORDER_TOKENS. Модели с рассуждениями тратят тот же потолок на размышление, поэтому запас
# большой. Величины не измерялись
ORDER_TOKEN_MARGIN = 2.0
MIN_ORDER_TOKENS = 4096


@dataclass
class Completion:
    """Ответ исполнителя вместе с расходом токенов. Исполнитель вправе вернуть и просто строку."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Почему поставщик закончил ответ: stop, length и т. п.
    finish_reason: str = "stop"


# Исполнитель: как получить ответ выбранного кандидата на сообщения диалога. Третий довод
# необязателен: словарь параметров хода (max_tokens, temperature), которые исполнитель передает
# поставщику. Исполнитель с двумя доводами его не получает
Executor = Callable[..., "Completion | str"]


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
    finish_reason: str = "stop"


class FaiRouter:
    """Роутер целиком. Вызывающему остается назвать кандидатов и сказать, как их спрашивать.
    Клиент модели для распознавания задания и разбора ответа принадлежит роутеру (довод llm); не
    задан, тогда общий Settings.llm."""

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
        seed: int | None = None,
        failure_cooldown: float = 300.0,
    ):
        if isinstance(bar, (int, float)) and not 0.0 <= bar <= 1.0:
            raise ValueError(f"Планка {bar} вне диапазона: нужна вероятность лайка от 0 до 1.")
        self.candidates = list(candidates)
        # Свой множитель температуры роутера для весов без своего множителя; None означает общий
        # из Settings. Фабрики ставят сюда пониженный Settings.PRIOR_TEMPERATURE_SCALE, когда все
        # кандидаты стартуют с начальных весов по рейтингам
        self.temperature_scale = temperature_scale
        # Планка достаточности на все ходы: число от 0 до 1 это обязательная вероятность лайка,
        # калибровка к ней подбирается по журналу человеческих отзывов; готовая SufficiencyBar
        # берется как есть. None означает выбор по метрике R без планки
        self.bar = bar
        # Сколько человеческих отзывов нужно, чтобы калибровка планки считалась осмысленной
        self.min_ratings = min_ratings
        # Профиль весов на все ходы: quality, balance, price либо свои RouteWeights; None означает
        # веса из Settings. Ход может назвать свой профиль и перекрыть этот
        self.route_weights = RouteWeights.profile(weights)
        self.topk = topk
        # Оценка ответа по форме и содержанию стоит двух обращений к модели на ход; ее можно отключить
        self.measure = measure
        self.llm = llm
        # Свой судья содержания нужен, например, для проверки фактов по вебу
        self.content_judge = content_judge or ContentJudge(llm)
        self.judge = Judge()
        # На сколько секунд отказавший кандидат (ошибка, таймаут, пустой ответ) выпадает из выбора.
        # Без паузы модель, отвечающая ошибкой сервера, выбиралась снова на каждом ходе
        self.failure_cooldown = failure_cooldown
        self._failed_until: dict[str, float] = {}
        self._execute = execute
        self._takes_options = _accepts(execute, 3)
        self._recognizer = None if llm is None else SpecInputService(llm)
        self._measurer = SpecOutputService(llm)
        # Генератор жребия свой у роутера: с seed выбор воспроизводим
        self._random = random.Random(seed)

        judge_model = self.content_judge.model
        if measure and judge_model is not None and any(candidate.name == judge_model for candidate in self.candidates):
            log.warning("Судья содержания %s среди кандидатов и будет судить сам себя; задайте ему другую модель "
                        "(content_judge).", judge_model)

        self.memory = RouterMemory(database_path, self.candidates, self.judge) if database_path else None
        if self.memory is not None:
            self.memory.load()

    @property
    def traces(self) -> "SqliteTraceStore | None":
        """Журнал ходов; None, если база не задана."""
        return None if self.memory is None else self.memory.traces

    @property
    def weights(self) -> "SqliteWeightsStore | None":
        """Хранилище весов; None, если база не задана."""
        return None if self.memory is None else self.memory.weights

    @classmethod
    def from_fractalrouter(
        cls,
        api_key: str,
        model_ids: "str | Iterable[str]" = catalog_module.POPULAR,
        database_path: str | None = None,
        prices: Prices | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = FRACTALROUTER_URL,
        **kwargs,
    ) -> "FaiRouter":
        """Роутер над моделями FractalRouter (fractalrouter.ru): цены и возможности из его
        каталога, ответы кандидатов и работа судьи через него же, одним ключом. Модели: список
        идентификаторов, «popular» (популярные из комплекта, по умолчанию) или «all» (весь каталог).
        Цены в рублях за миллион токенов; модели, которой в каталоге нет, цену задает словарь
        prices. Остальные доводы те же, что у from_openai_compatible."""
        return cls.from_openai_compatible(
            base_url, api_key, model_ids, database_path=database_path, prices=prices,
            judge_model=judge_model, catalog_fetch=lambda: catalog_module.fetch_fractalrouter(api_key), **kwargs)

    @classmethod
    def from_openrouter(
        cls,
        api_key: str,
        model_ids: "str | Iterable[str]" = catalog_module.POPULAR,
        database_path: str | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
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
        judge_model: str = DEFAULT_JUDGE_MODEL,
        tokens_per_second: dict[str, float] | None = None,
        benchmarks: "BenchmarkSnapshot | str | None" = None,
        timeout: float = 120.0,
        catalog_fetch: "Callable[[], Iterable[catalog_module.ModelInfo]] | None" = None,
        **kwargs,
    ) -> "FaiRouter":
        """Роутер над любым поставщиком по протоколу OpenAI chat completions: base_url это адрес
        вида https://host/v1, ключ уходит заголовком Bearer. Через того же поставщика и тем же
        ключом работает модель-судья judge_model; ее клиент принадлежит роутеру, а общий Settings.llm
        фабрика ставит, только если он еще не задан.

        Модели задаются списком идентификаторов либо именем набора: «popular» это популярные из
        комплекта (data/popular_models.json), которые есть в каталоге поставщика, «all» это весь его
        каталог. Профиль весов weights (quality, balance, price) действует на все ходы, как и
        планка достаточности bar: обязательная вероятность лайка от 0 до 1, калибруется по
        журналу человеческих отзывов.

        Цены кандидатов берутся из каталога catalog_fetch (по умолчанию каталог OpenRouter), а
        prices, (вход, выход) за миллион токенов в валюте поставщика, их задает или заменяет. Модель
        без цены ни там, ни там это ошибка: без цены роутеру нечего взвешивать. Каталог недоступен по
        сети, тогда цены берутся из комплекта пакета.

        Начальные веса кандидатов берутся из снимка внешних замеров (benchmarks: снимок, путь к
        нему или None для снимка из комплекта), поэтому роутер небесполезен с первого хода, а
        обучение на отзывах его уточняет в том же пространстве признаков: среднее задач кандидат
        хранит при себе, и ни загрузка, ни обучение его не меняют. Модель без рейтингов стартует с
        уровня поля. Пустая база на старте в порядке: веса в нее попадают при save(), ходы и отзывы
        при ask() и feedback()."""
        from fai_router import benchmarks as benchmarks_module
        from fai_router.training import benchmark_prior

        snapshot = benchmarks_module.resolve(benchmarks)
        if snapshot is None:
            log.warning("Снимок замеров не найден: кандидаты стартуют со случайных весов, "
                        "пока не накопятся отзывы.")
        # Снимок с чужими именами серий тише пустого: приора не получил бы никто, и роутер молча
        # стартовал бы с уровня поля, уверенный, что рейтинги у него есть
        if snapshot is not None and snapshot.entries and benchmark_prior.known_series(snapshot) == 0:
            raise ValueError("В снимке замеров нет ни одной серии из профилей: он собран другой версией "
                             "библиотеки. Пересоберите снимок или возьмите снимок из комплекта (benchmarks=None).")
        age = None if snapshot is None else snapshot.age_days
        if age is not None and age > SNAPSHOT_MAX_AGE_DAYS:
            log.warning("Снимку замеров %d дней, начальные веса устарели; пересоберите снимок.", int(age))

        prices = prices or {}
        # Список читается один раз: генератор иначе съедала бы проверка цен, и выбору доставался остаток
        requested = model_ids if isinstance(model_ids, str) else list(dict.fromkeys(model_ids))
        known: dict[str, catalog_module.ModelInfo] = {}
        if isinstance(requested, str) or any(model_id not in prices for model_id in requested):
            for model in catalog_module.fetch_or_popular(catalog_fetch or catalog_module.fetch):
                known.setdefault(model.id, model)
        model_list, _ = catalog_module.select(requested, known)
        if not model_list:
            raise ValueError("Список моделей пуст: роутеру не из кого выбирать.")
        candidates = catalog_module.create_candidates(model_list, known, prices, tokens_per_second, snapshot,
                                                      Settings.task_mean)

        clients: dict[str, OpenRouterClient] = {}

        def execute(candidate: RoutedElement, messages: Messages, options: dict[str, Any]) -> Completion:
            client = clients.setdefault(
                candidate.name, OpenRouterClient(api_key, candidate.name, timeout=timeout, base_url=base_url))
            data = client.complete_full(messages, temperature=options.get("temperature", 0.0),
                                        max_tokens=options.get("max_tokens"))
            prompt_tokens, completion_tokens = OpenRouterClient.usage(data)
            choice = (data.get("choices") or [{}])[0]
            return Completion((choice.get("message") or {}).get("content") or "", prompt_tokens, completion_tokens,
                              choice.get("finish_reason") or "stop")

        # Все кандидаты стартуют с начальных весов по рейтингам: долгая разведка не нужна, и
        # множитель температуры берется пониженный. Хотя бы один без рейтингов, тогда общий
        if all(candidate.prior_experience > 0 for candidate in candidates):
            kwargs.setdefault("temperature_scale", Settings.PRIOR_TEMPERATURE_SCALE)

        judge = OpenRouterClient(api_key, judge_model, timeout=timeout, base_url=base_url)
        # Общий клиент остается запасным для компонентов, собранных без клиента; чужой не затирается
        if Settings.llm is None:
            Settings.llm = judge
        kwargs.setdefault("llm", judge)
        return cls(candidates, execute, database_path, **kwargs)

    def ask(self, prompt: str, required: Capability = Capability.NONE,
            weights: "RouteWeights | str | None" = None,
            bar: "float | SufficiencyBar | None" = None,
            max_tokens: int | None = None, temperature: float | None = None) -> RouterAnswer:
        """Один ход по тексту запроса. Профиль весов на этот ход: quality, balance, price либо
        свои RouteWeights; не задан, тогда профиль роутера. Планка на этот ход: обязательная
        вероятность лайка либо готовая SufficiencyBar; не задана, тогда планка роутера. max_tokens и
        temperature уходят исполнителю."""
        return self.ask_messages([{"role": "user", "content": prompt}], required, weights, bar,
                                 max_tokens=max_tokens, temperature=temperature)

    def ask_messages(self, messages: Messages, required: Capability = Capability.NONE,
                     weights: "RouteWeights | str | None" = None,
                     bar: "float | SufficiencyBar | None" = None,
                     max_tokens: int | None = None, temperature: float | None = None) -> RouterAnswer:
        """Один ход по диалогу: задача распознается по последнему сообщению пользователя, а
        исполнителю уходит весь диалог целиком. content сообщения бывает строкой или массивом частей
        (текст, картинка): задачей считается текст частей. Объем входа для цены и окна контекста
        считается по всему диалогу: исполнитель читает его весь.

        max_tokens клиента уходит исполнителю как есть; не задан, тогда потолок следует из
        заказанного объема (вдвое больше заказа в токенах, не меньше MIN_ORDER_TOKENS и не больше
        предела ответа кандидата), а без заказанного объема действует умолчание поставщика.
        temperature клиента уходит как есть, без нее 0."""
        asked = [message for message in messages if str(message.get("role", "")).lower() == "user"]
        prompt = text_of(asked[-1].get("content")) if asked else ""
        if not prompt.strip():
            raise ValueError("В диалоге нет сообщения пользователя, задачу распознать не из чего.")

        input_tokens = sum(len(text_of(message.get("content"))) for message in messages) \
            / InputFeaturesService.EST_SYMBOL_PER_TOKEN
        trace = env.route(prompt, self._available(), self.topk, required,
                          weights=self.weights_for(weights), turns=len(asked),
                          bar=self.sufficiency_bar(bar), specs=self._recognizer,
                          input_tokens=input_tokens, rng=self._random)
        options = _options(trace.requested_spec, max_tokens, temperature)
        try:
            completion = env.execute(trace, lambda candidate: self._run(candidate, messages, options))
        finally:
            for name in trace.failed:
                self._failed_until[name] = time.monotonic() + self.failure_cooldown

        answer = RouterAnswer(completion.text, trace.winner.name, trace, trace.requested_spec,
                              prompt_tokens=completion.prompt_tokens,
                              completion_tokens=completion.completion_tokens,
                              reached=trace.bar_reached, finish_reason=completion.finish_reason)
        if self.measure and trace.requested_spec is not None:
            self._measure(answer, prompt)
        if self.memory is not None:
            answer.round_id = self.memory.append(trace, answer.actual, prompt, answer.assessment)
        return answer

    def weights_for(self, weights: "RouteWeights | str | None" = None) -> RouteWeights | None:
        """Веса хода: названные для хода, иначе веса роутера, иначе общие из Settings. Свой
        множитель температуры роутера подставляется в веса без своего множителя (веса по умолчанию
        и готовые профили quality, balance, price); веса с заданным множителем берутся как есть."""
        named = RouteWeights.profile(weights)
        chosen = self.route_weights if named is None else named
        if self.temperature_scale is None:
            return chosen
        resolved = Settings.current() if chosen is None else chosen
        return resolved if resolved.temperature_scale is not None else replace(
            resolved, temperature_scale=self.temperature_scale)

    def sufficiency_bar(self, bar: "float | SufficiencyBar | None" = None) -> SufficiencyBar | None:
        """Планка для хода; не задана, тогда планка роутера. Готовая SufficiencyBar берется как есть.
        Число это уровень, а калибровка к нему подбирается по журналу человеческих отзывов: пары
        «прогноз качества победителя в момент выбора и оценка человека», доля лайков как усадка для
        малоизученных. Пока отзывов меньше min_ratings, планки нет и ход идет по метрике R:
        калибровать не на чем, а планка без калибровки отсекала бы наугад. Калибровка запоминается и
        пересчитывается только после человеческого отзыва, обучения или загрузки."""
        bar = self.bar if bar is None else bar
        if bar is None or isinstance(bar, SufficiencyBar):
            return bar
        if not 0.0 <= bar <= 1.0:
            raise ValueError(f"Планка {bar} вне диапазона: нужна вероятность лайка от 0 до 1.")
        if self.memory is None:
            raise RuntimeError("Планка по уровню требует журнала: создайте роутер с database_path "
                               "либо передайте готовую SufficiencyBar с калибровкой.")
        state: CalibrationState = self.memory.current_calibration()
        if state.count < self.min_ratings:
            log.info("Человеческих отзывов %d из %d нужных: ход без планки.", state.count, self.min_ratings)
            return None
        return SufficiencyBar(bar, state.fit, prior_rate=state.rate)

    def calibration_pairs(self) -> list[tuple[float, float]]:
        """Пары для калибровки планки из журнала: прогноз качества победителя в момент выбора и
        оценка человека, последние тысяча человеческих отзывов. Автоотзывы не берутся: планка обещает
        вероятность лайка человека, а не согласие судьи с самим собой."""
        return self._require_memory().calibration_pairs()

    def feedback(self, round_id: int, score: float, human: bool = True) -> None:
        """Отзыв человека на ход: число от 0 до 1. Единица означает отличный ответ, ноль
        никуда не годный, 0,5 так себе. Отзыв перезаписывает автоотзыв судьи, на нем учатся и
        роутер, и судья; ход снова попадает в очередь обучения. Отзывом вне диапазона роутер не кормится."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Отзыв {score} вне диапазона: нужно число от 0 (плохо) до 1 (отлично).")
        self._require_memory().feedback(round_id, score, human)

    def train(self, epochs: int = 1) -> TrainingLoss:
        """Обучение по ходам журнала, которые еще не учили, в порядке записи. Возвращает ошибку
        последней эпохи раздельно у роутера и судьи (TrainingLoss, само значение это их сумма).

        Каждый ход учит один раз: прежде каждый вызов заново проходил тысячу последних ходов от уже
        обученных весов, и многократный train переобучал на них же. Ход считается обученным после
        save(): без сохранения веса пропали бы, а ход остался бы отмеченным. Новый отзыв возвращает
        ход в очередь. Среднее задач обучение не трогает: оно принадлежит вектору кандидата."""
        return self._require_memory().train(epochs)

    def save(self) -> None:
        """Сохраняет одной транзакцией векторы, которых касалось обучение (вместе с их средним
        задач), и матрицу судьи, затем отмечает обученные ходы. Необученные векторы не пишутся: при
        загрузке они затирали бы начальные веса по свежему снимку рейтингов."""
        self._require_memory().save()

    def load(self) -> None:
        """Восстанавливает веса со средним задач, матрицу судьи и опыт кандидатов. Вектор чужой
        размерности пропускается с предупреждением, остальные загружаются."""
        self._require_memory().load()

    @property
    def unsaved(self) -> bool:
        """Есть ли обученное в этом процессе и еще не сохраненное."""
        return self.memory is not None and self.memory.unsaved

    def _measure(self, answer: RouterAnswer, prompt: str) -> None:
        """Замер и оценка ответа. Форма и содержание не зависят друг от друга: замер структуры и суд
        содержания идут разом. Сбой судьи (сеть, поставщик, таймаут, неполный вердикт) ход не роняет:
        исполнитель уже ответил, и этот ответ дороже оценки. Ход без замера идет в журнал без
        автоотзыва. В журнал процесса пишется только тип ошибки: ее текст может нести запрос."""
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                measuring = pool.submit(self._measurer.get_specifications, answer.text)
                reviewing = pool.submit(self.content_judge.review, prompt, answer.requested, answer.text)
                actual, content = measuring.result(), reviewing.result()
            answer.score = self.judge.rate(answer.trace, answer.requested, actual)
            answer.critic = Judge.criticize(answer.requested, actual, content)
            answer.assessment = Judge.assess(answer.critic, content)
            answer.actual, answer.content = actual, content
        except Exception as error:  # noqa: BLE001 - любой сбой замера, ответ отдается без оценки
            log.warning("Ответ получен, но замерить его не удалось (%s).", type(error).__name__)
            answer.actual = answer.content = answer.score = answer.critic = answer.assessment = None

    def _run(self, candidate: RoutedElement, messages: Messages, options: dict[str, Any]) -> Completion:
        """Ответ кандидата; потолок ответа не выше предела кандидата, если тот известен."""
        if self._takes_options:
            own = dict(options)
            limit = int(candidate.context_limit / InputFeaturesService.EST_SYMBOL_PER_TOKEN)
            if limit > 0 and own.get("max_tokens") is not None:
                own["max_tokens"] = min(own["max_tokens"], limit)
            result = self._execute(candidate, messages, own)
        else:
            result = self._execute(candidate, messages)
        return Completion(result) if isinstance(result, str) else result

    def _available(self) -> list[RoutedElement]:
        """Кандидаты без недавнего отказа; отказали все, тогда все: пауза не повод остаться без исполнителя."""
        now = time.monotonic()
        healthy = [candidate for candidate in self.candidates if self._failed_until.get(candidate.name, 0.0) <= now]
        return healthy or list(self.candidates)

    def _require_memory(self) -> RouterMemory:
        if self.memory is None:
            raise RuntimeError("Журнал и хранилище не заданы: создайте роутер с database_path.")
        return self.memory


def text_of(content: Any) -> str:
    """Текст сообщения: строка как есть, у массива частей склеиваются текстовые части. Картинка и
    иное содержимое без текста задачей не считаются."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = [part if isinstance(part, str) else part.get("text") for part in content
             if isinstance(part, str) or (isinstance(part, dict) and part.get("type", "text") == "text")]
    return "\n".join(text for text in texts if isinstance(text, str))


def _options(requested: Specifications | None, max_tokens: int | None, temperature: float | None) -> dict[str, Any]:
    """Параметры исполнителя: заданные клиентом, а без них потолок по заказанному объему и температура 0."""
    if max_tokens is None and requested is not None and requested.symbol_length > 0:
        ordered = requested.symbol_length / InputFeaturesService.EST_SYMBOL_PER_TOKEN
        max_tokens = max(MIN_ORDER_TOKENS, math.ceil(ORDER_TOKEN_MARGIN * ordered))
    return {"max_tokens": max_tokens, "temperature": 0.0 if temperature is None else temperature}


def _accepts(function: Callable[..., Any], count: int) -> bool:
    """Принимает ли функция столько позиционных доводов."""
    try:
        inspect.signature(function).bind(*([None] * count))
    except (TypeError, ValueError):
        return False
    return True
