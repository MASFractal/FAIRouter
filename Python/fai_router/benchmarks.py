"""Снимок рейтингов моделей из внешних источников: арена (arena.ai) и Artificial Analysis.

Оба сайта отдают страницы сервером с данными внутри: в кусках ``self.__next_f.push([1,"..."])``
лежит JSON. Открытого API без ключа нет ни у одного, обычного GET достаточно. Ключ серии в снимке
начинается с источника: ``arena:text/coding``, ``aa:index/legal``.

    python -m fai_router.benchmarks --save data/benchmark-snapshot.json [--top 60]
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

USER_AGENT = "Mozilla/5.0 (FAIRouter)"

_CHUNK = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', re.S)
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
    """Строка рейтинга: модель и ее оценка в серии. Голоса есть только у арены."""

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
    каталога OpenRouter, арены и Artificial Analysis. «anthropic/claude-opus-4.7»,
    «claude-opus-4-7-high» и «claude-opus-4-7-20251101-high-32k» дают одно и то же, как и
    «anthropic/claude-haiku-4.5» с «claude-4-5-haiku-reasoning» (старый порядок имени у AA)."""
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
    """Часть ключа серии из названия источника: «Finance/Investing» становится «finance-investing»."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def rsc_payload(html: str) -> str:
    """Текст всех кусков RSC страницы подряд. Кусок это строковый литерал JS, правила
    экранирования у него те же, что у JSON."""
    parts = []
    for chunk in _CHUNK.findall(html):
        try:
            parts.append(json.loads(f'"{chunk}"'))
        except json.JSONDecodeError:
            parts.append(chunk.encode("utf-8").decode("unicode_escape", "ignore"))
    return "".join(parts)


def json_value_at(text: str, start: int) -> Any:
    """Значение JSON, начинающееся с позиции start."""
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


def get(url: str, timeout: float = 60.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_all(timeout: float = 60.0) -> BenchmarkSnapshot:
    """Снимок по обоим источникам."""
    from fai_router import analysis, arena

    entries = arena.fetch_all(timeout=timeout)
    entries.update(analysis.fetch_all(timeout=timeout))
    return BenchmarkSnapshot(datetime.now(timezone.utc).isoformat(), entries)


def _main(argv: list[str]) -> int:
    if "--save" not in argv:
        print(__doc__)
        return 2
    path = argv[argv.index("--save") + 1]
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 0
    snapshot = fetch_all()
    (snapshot.top(top) if top > 0 else snapshot).save(path)
    print(f"Сохранено серий: {len(snapshot.entries)}, файл {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
