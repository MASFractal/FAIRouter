"""Снимок внешних замеров качества моделей: серии оценок и сопоставление имен с каталогом.

Серия это одна шкала, по которой модели сравнивали снаружи. Ключ серии начинается с вида замера:
``pref:`` для парных предпочтений людей на задачах, ``bench:`` для прогонов бенчмарков. Библиотека
работает с готовым снимком; откуда он взят, ей знать не нужно (забор лежит вне репозитория).

Доля качества считается между худшим и лучшим в серии. Снимок в комплекте обрезан до первых строк
серии, и по обрезку размах предпочтений бывал 50 Эло: сильная модель в середине списка получала долю
0,15. Поэтому снимок может нести границы полной серии (поле "bounds": {"<серия>": {"low", "high",
"mean", "count"}}), их пишет top() перед обрезкой; нет границ, значит доля по строкам, как раньше.
Формат общий с версией на C#.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

_SEPARATORS = re.compile(r"[().,\s]+")
_VARIANT_TAIL = re.compile(r"^(\d{8}|\d{4}-\d{2}-\d{2}|\d{1,4}k|beta\d*)$")
_CLAUDE_OLD_ORDER = re.compile(r"^claude-(\d+(?:-\d+)?)-(haiku|sonnet|opus)(?=-|$)")

# Суффиксы имен, которыми источники различают одну и ту же модель: усилие размышления, режим,
# дата выпуска, размер контекста. Одна модель каталога совпадает со всеми такими вариантами
VARIANT_TOKENS = frozenset({
    "text", "high", "medium", "low", "minimal", "xhigh", "instant",
    "search", "grounding", "preview", "exp", "reasoning", "non",
})

# Суффиксы, которые у одних поставщиков означают режим той же модели (claude-opus-4-7-thinking,
# «GLM-5.2 (max)»), а у других отдельный продукт (qwen3.8-max, kimi-k2-thinking, gpt-5.1-codex-max).
# Точное имя их сохраняет, семейство срезает: сначала ищется точное совпадение, затем семейство.
# Исключение одно: у Claude размышление это режим той же модели каталога, отдельного продукта нет
PRODUCT_TOKENS = frozenset({"max", "thinking"})

_CLAUDE_THINKING = "thinking"


@dataclass(frozen=True)
class BenchmarkEntry:
    """Строка рейтинга: модель и ее оценка в серии. Голоса есть только у замеров предпочтений."""

    model_key: str
    display_name: str
    organization: str
    score: float
    votes: int = 0


@dataclass(frozen=True)
class SeriesBounds:
    """Границы полной серии, записанные до того, как снимок обрезали до первых строк: худшая и
    лучшая оценка, средняя (None, если не записана) и сколько строк было в серии."""

    low: float
    high: float
    mean: float | None = None
    count: int = 0


@dataclass
class BenchmarkSnapshot:
    """Рейтинги по сериям на один момент времени."""

    fetched_at: str
    entries: dict[str, list[BenchmarkEntry]]
    # Границы полных серий до обрезки; серии без записи считаются по своим строкам
    bounds: dict[str, SeriesBounds] | None = None
    # Индексы серий по каноническому имени: поиск зовут на каждую модель и каждую серию
    _indexes: dict[str, "_SeriesIndex"] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def age_days(self) -> float | None:
        """Возраст снимка в днях по fetched_at; дата не разбирается, значит None."""
        try:
            fetched = datetime.fromisoformat(self.fetched_at.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).total_seconds() / 86400

    def find(self, key: str, openrouter_id: str) -> BenchmarkEntry | None:
        """Запись модели каталога в серии; нет, значит None."""
        index = self._index(key)
        return None if index is None else index.find(openrouter_id)

    def value(self, key: str, openrouter_id: str) -> float | None:
        """Сама оценка модели в серии (скорость, доля рассуждений); None, если модели нет."""
        entry = self.find(key, openrouter_id)
        return None if entry is None else entry.score

    def quality(self, key: str, openrouter_id: str) -> float | None:
        """Качество модели в серии от 0 до 1: доля между худшим и лучшим в полной серии (по
        bounds, без них по строкам). Лидер получает единицу. Абсолютный смысл прогнозу дает
        калибровка, поэтому шкала условна."""
        index = self._index(key)
        entry = None if index is None else index.find(openrouter_id)
        return None if entry is None else index.share(entry.score)

    def shares(self, key: str) -> list[float]:
        """Качество каждой строки серии на той же шкале, что quality(); пусто, если серии нет."""
        index = self._index(key)
        return [] if index is None else [index.share(row.score) for row in self.entries[key]]

    def mean_share(self, key: str) -> tuple[float, int] | None:
        """Средняя доля качества в серии и сколько строк за ней стоит: по средней полной серии, если
        она записана, иначе по строкам снимка. Нет серии, значит None."""
        rows = self.entries.get(key) or []
        if not rows:
            return None
        index = self._index(key)
        bounds = (self.bounds or {}).get(key)
        if bounds is not None and bounds.mean is not None:
            return index.share(bounds.mean), max(bounds.count, len(rows))
        return sum(index.share(row.score) for row in rows) / len(rows), len(rows)

    def bounds_of(self, key: str) -> SeriesBounds | None:
        """Границы серии: записанные для полной серии, иначе по строкам; нет серии, значит None."""
        recorded = (self.bounds or {}).get(key)
        if recorded is not None:
            return recorded
        rows = self.entries.get(key) or []
        if not rows:
            return None
        scores = [row.score for row in rows]
        return SeriesBounds(min(scores), max(scores), sum(scores) / len(scores), len(rows))

    def top(self, count: int) -> "BenchmarkSnapshot":
        """Снимок, обрезанный до первых count строк каждой серии: для тестов и работы без сети.
        Границы полных серий записываются до обрезки, поэтому доля качества по обрезку та же, что по
        полной серии."""
        bounds = {key: self.bounds_of(key) for key in self.entries}
        return BenchmarkSnapshot(self.fetched_at, {key: rows[:count] for key, rows in self.entries.items()},
                                 {key: value for key, value in bounds.items() if value is not None})

    def save(self, path: str) -> None:
        data: dict[str, Any] = {"fetched_at": self.fetched_at,
                                "entries": {key: [asdict(row) for row in rows] for key, rows in self.entries.items()}}
        # Снимок без границ пишется без поля: формат обратно совместим
        if self.bounds is not None:
            data["bounds"] = {key: asdict(value) for key, value in self.bounds.items()}
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "BenchmarkSnapshot":
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        bounds = data.get("bounds")
        return cls(data["fetched_at"],
                   {key: [BenchmarkEntry(**row) for row in rows] for key, rows in data["entries"].items()},
                   None if bounds is None else {key: SeriesBounds(**value) for key, value in bounds.items()})

    def _index(self, key: str) -> "_SeriesIndex | None":
        """Индекс серии; список, выросший после построения, или новые границы индексируются заново."""
        rows = self.entries.get(key)
        if rows is None:
            return None
        bounds = (self.bounds or {}).get(key)
        index = self._indexes.get(key)
        if index is None or index.rows is not rows or index.count != len(rows) or index.bounds is not bounds:
            index = self._indexes[key] = _SeriesIndex(rows, bounds)
        return index


class _SeriesIndex:
    """Серия, разобранная один раз: лучшая запись на каждое каноническое имя по правилу find()
    (точное имя, затем семейство) и границы оценок для доли качества."""

    def __init__(self, rows: list[BenchmarkEntry], bounds: SeriesBounds | None):
        self.rows, self.count, self.bounds = rows, len(rows), bounds
        scores = [row.score for row in rows]
        self._low = bounds.low if bounds is not None else min(scores, default=0.0)
        self._high = bounds.high if bounds is not None else max(scores, default=0.0)
        self._exact: dict[str, BenchmarkEntry] = {}
        self._family: dict[str, BenchmarkEntry] = {}
        for row in rows:
            _add(self._exact, row, relaxed=False)
            _add(self._family, row, relaxed=True)

    def find(self, openrouter_id: str) -> BenchmarkEntry | None:
        return self._exact.get(canonical(openrouter_id)) or self._family.get(canonical(openrouter_id, relaxed=True))

    def share(self, score: float) -> float:
        return 1.0 if self._high <= self._low else min(max((score - self._low) / (self._high - self._low), 0.0), 1.0)


@lru_cache(maxsize=20000)
def canonical(name: str, relaxed: bool = False) -> str:
    """Имя модели без поставщика, вариантов усилия, режима, даты и контекста: то, что совпадает у
    каталога OpenRouter и внешних замеров. «anthropic/claude-opus-4.7», «claude-opus-4-7-high» и
    «claude-opus-4-7-20251101-high-32k» дают одно и то же, как и «anthropic/claude-haiku-4.5» с
    «claude-4-5-haiku-reasoning» (иной порядок слов в имени).

    Хвосты max и thinking точное имя сохраняет: qwen3.8-max и qwen3.8, gpt-5.1-codex-max и
    gpt-5.1-codex это разные продукты, и прежде они склеивались в одно имя. Семейство (relaxed)
    срезает и их: так модель каталога находит свою строку, когда в серии есть только вариант с режимом.
    Правило общее с версией на C# (ModelNames.Canonical)."""
    text = name.strip().lower()
    text = text.split("/", 1)[-1]
    text = text.split(":", 1)[0]
    tokens = _SEPARATORS.sub("-", text).strip("-").split("-")
    claude = tokens[0] == "claude"
    while len(tokens) > 1 and (_is_variant(tokens[-1], relaxed) or (claude and tokens[-1] == _CLAUDE_THINKING)):
        tokens.pop()
    # Дата вида 2024-05-13 после разбиения по дефису стала тремя числами
    while len(tokens) > 3 and _is_digits(tokens[-3], 4) and _is_digits(tokens[-2], 2) and _is_digits(tokens[-1], 2):
        del tokens[-3:]
    return _CLAUDE_OLD_ORDER.sub(r"claude-\2-\1", "-".join(token for token in tokens if token))


def find(openrouter_id: str, entries: Iterable[BenchmarkEntry]) -> BenchmarkEntry | None:
    """Запись для модели каталога: среди совпавших по точному каноническому имени берется та, у
    которой больше голосов, а при равных голосах самая короткая, то есть вариант без суффиксов.
    Точных нет, тогда так же среди совпавших по семейству."""
    rows = list(entries)
    return _best(rows, canonical(openrouter_id), False) or _best(rows, canonical(openrouter_id, relaxed=True), True)


def slug(name: str) -> str:
    """Часть ключа серии из названия замера: «Finance/Investing» становится «finance-investing»."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Где искать снимок, если его не назвали: в комплекте пакета; второй путь на случай старой раскладки
_DEFAULT_PATHS = (
    Path(__file__).resolve().parent / "data" / "benchmark-snapshot.json",
    Path(__file__).resolve().parents[2] / "data" / "benchmark-snapshot.json",
)


def default_snapshot() -> BenchmarkSnapshot | None:
    """Снимок из комплекта: fai_router/data/benchmark-snapshot.json в пакете. None,
    если файла нет ни там, ни там, и тогда кандидаты стартуют со случайных весов."""
    for path in _DEFAULT_PATHS:
        if path.is_file():
            return BenchmarkSnapshot.load(str(path))
    return None


def resolve(source: "BenchmarkSnapshot | str | None") -> BenchmarkSnapshot | None:
    """Снимок по тому, что передали: сам снимок, путь к файлу или None для снимка из комплекта."""
    if source is None:
        return default_snapshot()
    if isinstance(source, str):
        return BenchmarkSnapshot.load(source)
    return source


def _better(row: BenchmarkEntry, than: BenchmarkEntry) -> bool:
    """Запись лучше другой той же модели: больше голосов, при равных короче ключ; при полном
    равенстве остается прежняя."""
    return row.votes > than.votes or (row.votes == than.votes and len(row.model_key) < len(than.model_key))


def _best(rows: list[BenchmarkEntry], wanted: str, relaxed: bool) -> BenchmarkEntry | None:
    best = None
    for row in rows:
        if (canonical(row.display_name, relaxed) == wanted or canonical(row.model_key, relaxed) == wanted) \
                and (best is None or _better(row, best)):
            best = row
    return best


def _add(index: dict[str, BenchmarkEntry], row: BenchmarkEntry, relaxed: bool) -> None:
    for name in dict.fromkeys((canonical(row.display_name, relaxed), canonical(row.model_key, relaxed))):
        best = index.get(name)
        if best is None or _better(row, best):
            index[name] = row


def _is_variant(token: str, relaxed: bool) -> bool:
    return token in VARIANT_TOKENS or bool(_VARIANT_TAIL.fullmatch(token)) or (relaxed and token in PRODUCT_TOKENS)


def _is_digits(token: str, length: int) -> bool:
    return len(token) == length and token.isascii() and token.isdigit()
