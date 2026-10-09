"""Каталог моделей: цены, размеры окна и возможности берутся у поставщика, у FractalRouter или
OpenRouter. Скорость поставщики не публикуют, поэтому число токенов в секунду задает вызывающий.

Цены хранятся в валюте поставщика за миллион токенов: у OpenRouter доллары, у FractalRouter
рубли. Для выбора важны только отношения цен кандидатов между собой, поэтому валюта роли не
играет, лишь бы все кандидаты одного роутера были из одного каталога.

Отрицательная цена означает, что она неизвестна: поставщик ее не сообщил, сообщил непонятно или сам
пометил минус единицей (так у OpenRouter помечен openrouter/auto). Ноль означает «бесплатно», и
путать их нельзя: прежде цена без данных становилась нулем, и модель выигрывала как бесплатная."""

from __future__ import annotations

import http.client
import json
import logging
import math
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from fai_router.enums import Capability
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import GLOBAL_MEAN, Settings

if TYPE_CHECKING:
    import numpy as np

    from fai_router.benchmarks import BenchmarkSnapshot

log = logging.getLogger("fai_router")

OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
FRACTALROUTER_MODELS = "https://api.fractalrouter.ru/v1/models"

# Цена, которой нет: поставщик ее не сообщил
UNKNOWN_PRICE = -1.0


@dataclass(frozen=True)
class ModelInfo:
    id: str
    title: str
    dollars_per_million_input: float
    dollars_per_million_output: float
    context_tokens: int
    max_answer_tokens: int
    capabilities: Capability


# Скорость модели, о которой ничего не известно и снимка замеров нет, токенов в секунду. Со снимком
# умолчанием служит нижняя граница серии скорости
DEFAULT_TOKENS_PER_SECOND = 50.0

# Именованные наборы моделей вместо списка идентификаторов: весь каталог поставщика либо
# список популярных из комплекта (data/popular_models.json)
ALL = "all"
POPULAR = "popular"
POPULAR_MODELS_PATH = Path(__file__).resolve().parent / "data" / "popular_models.json"


def fetch(timeout: float = 30.0) -> list[ModelInfo]:
    """Загружает каталог OpenRouter, ключ не нужен."""
    with urllib.request.urlopen(OPENROUTER_MODELS, timeout=timeout) as response:
        return parse(response.read().decode("utf-8"))


def fetch_fractalrouter(api_key: str, timeout: float = 30.0) -> list[ModelInfo]:
    """Загружает каталог FractalRouter; он отдается только по ключу. Цены в рублях."""
    request = urllib.request.Request(FRACTALROUTER_MODELS, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_fractalrouter(response.read().decode("utf-8"))


def fetch_or_popular(fetch_catalog: Callable[[], Iterable[ModelInfo]]) -> list[ModelInfo]:
    """Каталог поставщика, а при сбое сети, таймауте или нечитаемом ответе популярные модели из
    комплекта с ценами из него (доллары OpenRouter на дату сборки). Без сети роутер прежде не
    создавался вовсе."""
    try:
        return list(fetch_catalog())
    except (OSError, ValueError, http.client.HTTPException) as error:
        log.warning("Каталог поставщика недоступен (%s); цены взяты из комплекта пакета.", type(error).__name__)
        return popular_model_infos()


def popular_models() -> list[str]:
    """Идентификаторы популярных моделей из комплекта: модели с внешними рейтингами в снимке
    замеров и недорогие рабочие лошадки. Это выбор по умолчанию, когда модели не названы."""
    return [model.id for model in popular_model_infos()]


def popular_model_infos() -> list[ModelInfo]:
    """Популярные модели из комплекта со сведениями: цены в долларах за миллион токенов на дату
    сборки (нет цены, значит неизвестна). Окно и возможности комплект не хранит."""
    with open(POPULAR_MODELS_PATH, encoding="utf-8") as file:
        items = json.load(file).get("models") or []
    return [ModelInfo(_text(item, "id"), _text(item, "title") or "",
                      _price(item.get("usd_per_million_input")), _price(item.get("usd_per_million_output")),
                      0, 0, Capability.CODE | Capability.FORMULAS | Capability.TOOLS)
            for item in items if isinstance(item, dict) and _text(item, "id")]


def select(model_ids: "str | Iterable[str]", known: dict[str, ModelInfo]) -> tuple[list[str], bool]:
    """Список идентификаторов по тому, что передали: строка «all» дает весь каталог поставщика с
    известной ценой, «popular» дает популярные из комплекта, которые есть в каталоге, список берется
    как есть, без повторов. Второе значение говорит, обязан ли каждый идентификатор найтись: у
    именованного набора отсутствующие модели молча пропускаются, у списка их отсутствие это ошибка."""
    if isinstance(model_ids, str):
        name = model_ids.strip().lower()
        if name == ALL:
            return [model_id for model_id, model in known.items() if _priced(model)], False
        if name == POPULAR:
            wanted = popular_models()
            found = [model_id for model_id in wanted if model_id in known]
            if len(found) < len(wanted):
                log.info("Популярных моделей в каталоге поставщика %d из %d.", len(found), len(wanted))
            if not found:
                raise ValueError("Ни одной популярной модели в каталоге поставщика нет: назовите модели списком.")
            return found, False
        raise ValueError(f"Неизвестный набор моделей «{model_ids}»: есть all и popular, либо список идентификаторов.")
    return list(dict.fromkeys(model_ids)), True


def load(path: str) -> list[ModelInfo]:
    with open(path, encoding="utf-8") as file:
        return [_from_dict(item) for item in json.load(file)]


def save(path: str, models: Iterable[ModelInfo]) -> None:
    """Снимок каталога, чтобы состав и цены не менялись между прогонами."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump([_to_dict(model) for model in models], file, ensure_ascii=False, indent=2)


def create_element(model: ModelInfo, tokens_per_second: float | None = None,
                   benchmarks: BenchmarkSnapshot | None = None,
                   task_mean: "np.ndarray | None | object" = GLOBAL_MEAN) -> RoutedElement:
    """Кандидат на исполнение из сведений каталога. Предел ответа переводится из токенов в
    символы, потому что context_limit кандидата измеряется в символах; окно остается в токенах.

    Со снимком рейтингов кандидат стартует не со случайного вектора, а с прогноза по сериям внешних
    замеров (benchmark_prior), и это стоит условного опыта (prior_experience). Модель, которой в
    рейтингах нет, получает уровень поля (field_quality) во всех сериях: прежде она получала
    случайный вектор и побеждала или проигрывала наугад. Оттуда же берутся скорость, если ее не
    назвали (нет в серии, значит нижняя граница серии), и поправка цены на рассуждения. Вектор
    строится в пространстве среднего task_mean (не задано, значит общее Settings.task_mean)."""
    # Импорт здесь, а не наверху: профили рейтингов подгружает только тот, кто их просит
    from fai_router.training import benchmark_prior

    mean = Settings.task_mean if task_mean is GLOBAL_MEAN else task_mean
    speed = tokens_per_second
    if speed is None and benchmarks is not None:
        speed = (benchmark_prior.tokens_per_second(benchmarks, model.id)
                 or benchmark_prior.default_tokens_per_second(benchmarks))
    element = RoutedElement(
        name=model.id,
        tps=DEFAULT_TOKENS_PER_SECOND if speed is None else speed,
        dpmt_inp=_known_or_unknown(model.dollars_per_million_input),
        dpmt_outp=_known_or_unknown(model.dollars_per_million_output),
        capabilities=model.capabilities,
        context_limit=int(max(model.max_answer_tokens, 0) * InputFeaturesService.EST_SYMBOL_PER_TOKEN),
        context_window=max(model.context_tokens, 0),
        task_mean=mean,
    )
    if benchmarks is None:
        return element
    prior = benchmark_prior.vector(benchmarks, model.id, mean)
    if prior is not None:
        element.ideal_match_vector = prior
        element.prior_experience = Settings.PRIOR_EXPERIENCE
    else:
        field = benchmark_prior.field_quality(benchmarks)
        if field is not None:
            element.ideal_match_vector = benchmark_prior.uniform(field, mean)
    ratio = benchmark_prior.cost_ratio(benchmarks, model.id)
    if ratio is not None:
        element.cost_ratio = ratio
    return element


def create_candidates(ids: Iterable[str], known: dict[str, ModelInfo],
                      prices: dict[str, tuple[float, float]] | None = None,
                      tokens_per_second: dict[str, float] | None = None,
                      benchmarks: BenchmarkSnapshot | None = None,
                      task_mean: "np.ndarray | None | object" = GLOBAL_MEAN) -> list[RoutedElement]:
    """Кандидаты по идентификаторам: цена из prices поверх каталога, остальное из каталога. Модель
    только из prices считается способной на все, кроме того, что опровергает каталог; модели без цены
    ни там, ни там быть не может. Повторы в списке пропускаются."""
    prices, speeds = prices or {}, tokens_per_second or {}
    candidates = []
    for model_id in dict.fromkeys(ids):
        found = known.get(model_id)
        if model_id in prices:
            inp, outp = prices[model_id]
            base = found or ModelInfo(model_id, model_id, 0.0, 0.0, 0, 0, Capability.ALL)
            info = replace(base, dollars_per_million_input=float(inp), dollars_per_million_output=float(outp))
        elif found is not None:
            info = found
        else:
            raise ValueError(
                f"У модели {model_id} нет цены в prices, и в каталоге поставщика ее нет. "
                f"Задайте цену: prices={{\"{model_id}\": (вход, выход)}} за миллион токенов.")
        if not _priced(info):
            log.warning("Цена модели %s неизвестна, в выборе она получает худшую цену группы.", model_id)
        candidates.append(create_element(info, speeds.get(model_id), benchmarks, task_mean))
    return candidates


def parse(text: str) -> list[ModelInfo]:
    """Разбирает ответ каталога OpenRouter. Битая запись пропускается, а не роняет весь каталог;
    нет цены или она отрицательна, значит цена неизвестна."""
    models = []
    for item in _items(json.loads(text), "data"):
        model_id = _text(item, "id")
        if not model_id:
            continue
        pricing = _object(item.get("pricing"))
        models.append(ModelInfo(
            id=model_id,
            title=_text(item, "name") or "",
            dollars_per_million_input=_per_million(pricing.get("prompt")),
            dollars_per_million_output=_per_million(pricing.get("completion")),
            context_tokens=_count(item.get("context_length")),
            max_answer_tokens=_count(_object(item.get("top_provider")).get("max_completion_tokens")),
            capabilities=_capabilities(item, pricing),
        ))
    return models


def parse_fractalrouter(text: str) -> list[ModelInfo]:
    """Каталог FractalRouter: цены уже за миллион токенов, в рублях. Предел ответа каталог не
    сообщает, поэтому ограничения по объему у кандидата нет. Битая запись пропускается."""
    payload = json.loads(text)
    models = []
    for item in _items(payload, None if isinstance(payload, list) else "data"):
        model_id = _text(item, "id")
        if not model_id:
            continue
        pricing = _object(item.get("pricing"))
        # Умение вызывать инструменты каталог не сообщает; считаем, что умеют все, иначе запрос
        # с required=Capability.TOOLS не нашел бы ни одного кандидата
        capabilities = Capability.CODE | Capability.FORMULAS | Capability.TOOLS
        if {"vision", "image"} & set(_strings(item.get("modalities"))):
            capabilities |= Capability.VISION
        models.append(ModelInfo(
            id=model_id,
            title=_text(item, "name") or model_id,
            dollars_per_million_input=_price(pricing.get("prompt_rub_per_1m")),
            dollars_per_million_output=_price(pricing.get("completion_rub_per_1m")),
            context_tokens=_count(item.get("context_length")),
            max_answer_tokens=_count(item.get("max_completion_tokens")),
            capabilities=capabilities,
        ))
    return models


def _priced(model: ModelInfo) -> bool:
    # Служебные записи с неизвестной ценой набору «все» не нужны
    return model.dollars_per_million_input >= 0 and model.dollars_per_million_output >= 0


def _capabilities(item: dict[str, Any], pricing: dict[str, Any]) -> Capability:
    # Писать код и формулы умеет любая текстовая модель, это не отличительная черта
    capabilities = Capability.CODE | Capability.FORMULAS
    if "image" in _strings(_object(item.get("architecture")).get("input_modalities")):
        capabilities |= Capability.VISION
    if "tools" in _strings(item.get("supported_parameters")):
        capabilities |= Capability.TOOLS
    # Свой поиск в сети у модели есть, если поставщик назначил ему цену
    search = _decimal(pricing.get("web_search"))
    if search is not None and search > 0:
        capabilities |= Capability.WEB_SEARCH
    return capabilities


def _decimal(value: Any) -> float | None:
    """Конечное число из числа или строки (строки бывают с запятой вместо точки); иначе None."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(str(value).replace(",", "."))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _price(value: Any) -> float:
    """Цена за миллион; нет ее или она отрицательна, значит неизвестна."""
    number = _decimal(value)
    return number if number is not None and number >= 0 else UNKNOWN_PRICE


def _per_million(value: Any) -> float:
    """Цены OpenRouter приходят строками и за один токен; нет цены или она отрицательна, значит неизвестна."""
    number = _decimal(value)
    return number * 1e6 if number is not None and number >= 0 else UNKNOWN_PRICE


def _known_or_unknown(price: float) -> float:
    # Отрицательная цена любого вида приводится к одной метке «неизвестна»
    return price if math.isfinite(price) and price >= 0 else UNKNOWN_PRICE


def _count(value: Any) -> int:
    number = _decimal(value)
    return 0 if number is None else int(min(max(number, 0), 2 ** 31 - 1))


def _text(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if isinstance(value, str):
        return value
    return str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(root: Any, key: str | None) -> list[dict[str, Any]]:
    items = root if key is None else _object(root).get(key)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _to_dict(model: ModelInfo) -> dict[str, Any]:
    data = asdict(model)
    data["capabilities"] = int(model.capabilities)
    return data


def _from_dict(data: dict[str, Any]) -> ModelInfo:
    data = dict(data)
    data["capabilities"] = Capability(data["capabilities"])
    return ModelInfo(**data)
