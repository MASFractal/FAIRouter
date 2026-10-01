"""Снимок внешних замеров качества моделей: серии оценок и сопоставление имен с каталогом.

Серия это одна шкала, по которой модели сравнивали снаружи. Ключ серии начинается с вида замера:
``pref:`` для парных предпочтений людей на задачах, ``bench:`` для прогонов бенчмарков. Библиотека
работает с готовым снимком; откуда он взят, ей знать не нужно (забор лежит вне репозитория).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_SEPARATORS = re.compile(r"[().,\s]+")
_VARIANT_TAIL = re.compile(r"^(\d{8}|\d{4}-\d{2}-\d{2}|\d{1,4}k|beta\d*)$")
_CLAUDE_OLD_ORDER = re.compile(r"^claude-(\d+(?:-\d+)?)-(haiku|sonnet|opus)(?=-|$)")

# Суффиксы имен, которыми источники различают одну и ту же модель: усилие размышления, режим,
# дата выпуска, размер контекста. Одна модель каталога совпадает со всеми такими вариантами
VARIANT_TOKENS = frozenset({
    "text", "high", "max", "medium", "low", "minimal", "xhigh", "instant", "thinking",
    "search", "grounding", "preview", "exp", "reasoning", "non",
})


@dataclass(frozen=True)
class BenchmarkEntry:
    """Строка рейтинга: модель и ее оценка в серии. Голоса есть только у замеров предпочтений."""

    model_key: str
    display_name: str
    organization: str
    score: float
    votes: int = 0


@dataclass
class BenchmarkSnapshot:
    """Рейтинги по сериям на один момент времени."""

    fetched_at: str
    entries: dict[str, list[BenchmarkEntry]]

    def find(self, key: str, openrouter_id: str) -> BenchmarkEntry | None:
        return find(openrouter_id, self.entries.get(key, []))

    def value(self, key: str, openrouter_id: str) -> float | None:
        """Сама оценка модели в серии (скорость, доля рассуждений); None, если модели нет."""
        entry = self.find(key, openrouter_id)
        return None if entry is None else entry.score

    def quality(self, key: str, openrouter_id: str) -> float | None:
        """Качество модели в серии от 0 до 1: доля между худшим и лучшим. Лидер получает единицу.
        Абсолютный смысл прогнозу дает калибровка, поэтому шкала условна."""
        rows = self.entries.get(key, [])
        entry = find(openrouter_id, rows)
        if entry is None:
            return None
        low = min(row.score for row in rows)
        high = max(row.score for row in rows)
        return 1.0 if high <= low else (entry.score - low) / (high - low)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"fetched_at": self.fetched_at,
                       "entries": {key: [asdict(row) for row in rows] for key, rows in self.entries.items()}},
                      file, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "BenchmarkSnapshot":
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return cls(data["fetched_at"],
                   {key: [BenchmarkEntry(**row) for row in rows] for key, rows in data["entries"].items()})

    def top(self, count: int) -> "BenchmarkSnapshot":
        """Снимок, обрезанный до первых count строк каждой серии: для тестов и работы без сети."""
        return BenchmarkSnapshot(self.fetched_at, {key: rows[:count] for key, rows in self.entries.items()})


def canonical(name: str) -> str:
    """Имя модели без поставщика, вариантов усилия, режима, даты и контекста: то, что совпадает у
    каталога OpenRouter и внешних замеров. «anthropic/claude-opus-4.7»,
    «claude-opus-4-7-high» и «claude-opus-4-7-20251101-high-32k» дают одно и то же, как и
    «anthropic/claude-haiku-4.5» с «claude-4-5-haiku-reasoning» (иной порядок слов в имени)."""
    text = name.strip().lower()
    text = text.split("/", 1)[-1]
    text = text.split(":", 1)[0]
    text = _SEPARATORS.sub("-", text).strip("-")
    tokens = text.split("-")
    while len(tokens) > 1 and (tokens[-1] in VARIANT_TOKENS or _VARIANT_TAIL.fullmatch(tokens[-1])):
        tokens.pop()
    # Дата вида 2024-05-13 после разбиения по дефису стала тремя числами
    while len(tokens) > 3 and re.fullmatch(r"\d{4}", tokens[-3]) and re.fullmatch(r"\d{2}", tokens[-2]) \
            and re.fullmatch(r"\d{2}", tokens[-1]):
        del tokens[-3:]
    return _CLAUDE_OLD_ORDER.sub(r"claude-\2-\1", "-".join(token for token in tokens if token))


def find(openrouter_id: str, entries: Iterable[BenchmarkEntry]) -> BenchmarkEntry | None:
    """Запись для модели каталога: среди совпавших по каноническому имени берется та, у которой
    больше голосов, а при равных голосах самая короткая, то есть вариант без суффиксов."""
    wanted = canonical(openrouter_id)
    matches = [row for row in entries
               if canonical(row.display_name) == wanted or canonical(row.model_key) == wanted]
    return max(matches, key=lambda row: (row.votes, -len(row.model_key))) if matches else None


def slug(name: str) -> str:
    """Часть ключа серии из названия замера: «Finance/Investing» становится «finance-investing»."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Где искать снимок, если его не назвали: в комплекте пакета; второй путь на случай старой раскладки
_DEFAULT_PATHS = (
    Path(__file__).resolve().parent / "data" / "benchmark-snapshot.json",
    Path(__file__).resolve().parents[2] / "data" / "benchmark-snapshot.json",
)


def default_snapshot() -> BenchmarkSnapshot | None:
    """Снимок из комплекта: data/benchmark-snapshot.json в пакете либо в репозитории. None,
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
