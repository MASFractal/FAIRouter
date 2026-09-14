"""Каталог моделей: цены, размеры окна и возможности берутся у OpenRouter. Скорость поставщик
не публикует, поэтому число токенов в секунду задает вызывающий."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable

from fai_router.enums import Capability
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService

if TYPE_CHECKING:
    from fai_router.benchmarks import BenchmarkSnapshot

OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    title: str
    dollars_per_million_input: float
    dollars_per_million_output: float
    context_tokens: int
    max_answer_tokens: int
    capabilities: Capability
    intelligence_index: float
    coding_index: float
    agentic_index: float


# Скорость модели, о которой ничего не известно, токенов в секунду
DEFAULT_TOKENS_PER_SECOND = 50.0


def fetch(timeout: float = 60.0) -> list[ModelInfo]:
    """Загружает каталог у поставщика, ключ не нужен."""
    with urllib.request.urlopen(OPENROUTER_MODELS, timeout=timeout) as response:
        return parse(response.read().decode("utf-8"))


def load(path: str) -> list[ModelInfo]:
    with open(path, encoding="utf-8") as file:
        return [_from_dict(item) for item in json.load(file)]


def save(path: str, models: Iterable[ModelInfo]) -> None:
    """Снимок каталога, чтобы состав и цены не менялись между прогонами."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump([_to_dict(model) for model in models], file, ensure_ascii=False, indent=2)


def create_element(model: ModelInfo, tokens_per_second: float | None = None,
                   benchmarks: BenchmarkSnapshot | None = None) -> RoutedElement:
    """Кандидат на исполнение из сведений каталога. Предел ответа переводится из токенов в
    символы, потому что context_limit кандидата измеряется в символах. Со снимком рейтингов
    кандидат стартует не со случайного вектора, а с прогноза по сериям арены и Artificial Analysis
    (benchmark_prior); оттуда же берутся скорость, если ее не назвали, и поправка цены на
    рассуждения. Модели, которой в рейтингах нет, снимок не касается."""
    # Импорт здесь, а не наверху: профили рейтингов подгружает только тот, кто их просит
    from fai_router.training import benchmark_prior

    speed = tokens_per_second
    if speed is None and benchmarks is not None:
        speed = benchmark_prior.tokens_per_second(benchmarks, model.id)
    element = RoutedElement(
        name=model.id,
        tps=DEFAULT_TOKENS_PER_SECOND if speed is None else speed,
        dpmt_inp=model.dollars_per_million_input,
        dpmt_outp=model.dollars_per_million_output,
        capabilities=model.capabilities,
        context_limit=int(model.max_answer_tokens * InputFeaturesService.EST_SYMBOL_PER_TOKEN),
    )
    if benchmarks is None:
        return element
    prior = benchmark_prior.vector(benchmarks, model.id)
    if prior is not None:
        element.ideal_match_vector = prior
    ratio = benchmark_prior.cost_ratio(benchmarks, model.id)
    if ratio is not None:
        element.cost_ratio = ratio
    return element

def parse(text: str) -> list[ModelInfo]:
    models = []
    for item in json.loads(text)["data"]:
        pricing = item.get("pricing") or {}
        architecture = item.get("architecture") or {}
        provider = item.get("top_provider") or {}
        analysis = (item.get("benchmarks") or {}).get("artificial_analysis") or {}
        models.append(ModelInfo(
            id=item.get("id", ""),
            title=item.get("name", ""),
            dollars_per_million_input=_per_million(pricing.get("prompt")),
            dollars_per_million_output=_per_million(pricing.get("completion")),
            context_tokens=int(item.get("context_length") or 0),
            max_answer_tokens=int(provider.get("max_completion_tokens") or 0),
            capabilities=_capabilities(item, architecture),
            intelligence_index=float(analysis.get("intelligence_index") or 0),
            coding_index=float(analysis.get("coding_index") or 0),
            agentic_index=float(analysis.get("agentic_index") or 0),
        ))
    return models


def _per_million(value: Any) -> float:
    # Цены поставщик отдает строками и за один токен
    try:
        return float(value) * 1e6
    except (TypeError, ValueError):
        return 0.0


def _capabilities(item: dict[str, Any], architecture: dict[str, Any]) -> Capability:
    # Писать код и формулы умеет любая текстовая модель. Поиск в сети из каталога не выводится
    capabilities = Capability.CODE | Capability.FORMULAS
    if "image" in (architecture.get("input_modalities") or []):
        capabilities |= Capability.VISION
    if "tools" in (item.get("supported_parameters") or []):
        capabilities |= Capability.TOOLS
    return capabilities


def _to_dict(model: ModelInfo) -> dict[str, Any]:
    data = asdict(model)
    data["capabilities"] = int(model.capabilities)
    return data


def _from_dict(data: dict[str, Any]) -> ModelInfo:
    data = dict(data)
    data["capabilities"] = Capability(data["capabilities"])
    return ModelInfo(**data)
