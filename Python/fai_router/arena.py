"""Рейтинги арены (arena.ai): реестр категорий и разбор страниц.

Страница категории отдает JSON рейтинга внутри кусков RSC:
``"id":"leaderboard-sets/public/leaderboards/<ключ>/leaderboard-snapshots/latest","entries":[...]``.
Рейтинг фактологии отдельной страницей не отдается: переключатель веса фактологии работает в
браузере, адрес ``text/overall-factuality`` возвращает общий рейтинг без контроля стиля. Поэтому
фактология берется у Artificial Analysis (analysis.py)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from fai_router.benchmarks import BenchmarkEntry, get, json_value_at, rsc_payload

ARENA_URL = "https://arena.ai/leaderboard/{arena}/{name}"

_ENTRIES = re.compile(r'leaderboards/([a-z0-9_\-]+)/leaderboard-snapshots/latest","entries":')


@dataclass(frozen=True)
class ArenaCategory:
    """Категория арены: арена (text, code, search), имя категории из адреса и часть ключа
    рейтинга, если она не совпадает с именем."""

    arena: str
    name: str
    slug: str | None = None

    @property
    def key(self) -> str:
        return f"arena:{self.arena}/{self.name}"

    @property
    def url(self) -> str:
        return ARENA_URL.format(arena=self.arena, name=self.name)

    @property
    def expected_key_part(self) -> str:
        return self.slug or self.name.replace("-", "_")


# Все 29 текстовых категорий арены, кроме служебной exclude-ties. Список повторяет реестр
# страницы leaderboard/text
TEXT_CATEGORIES = (
    "overall", "expert",
    "industry-software-and-it-services", "industry-writing-and-literature-and-language",
    "industry-life-and-physical-and-social-science", "industry-entertainment-and-sports-and-media",
    "industry-business-and-management-and-financial-operations", "industry-mathematical",
    "industry-legal-and-government", "industry-medicine-and-healthcare",
    "math", "instruction-following", "multi-turn", "creative-writing", "coding",
    "hard-prompts", "hard-prompts-english", "longer-query",
    "english", "non-english", "chinese", "french", "german", "spanish", "russian", "japanese",
    "korean", "polish",
)

# Текстовые категории, пять категорий арены кода и поисковая арена
CATEGORIES = (
    *(ArenaCategory("text", name) for name in TEXT_CATEGORIES),
    ArenaCategory("code", "frontend"),
    ArenaCategory("code", "fullstack"),
    ArenaCategory("code", "html"),
    ArenaCategory("code", "react"),
    ArenaCategory("code", "brand-marketing"),
    ArenaCategory("search", "overall", "search-overall"),
)


def parse(html: str) -> tuple[str, list[BenchmarkEntry]]:
    """Ключ рейтинга и его записи со страницы арены. Страница без рейтинга дает ValueError."""
    text = rsc_payload(html)
    match = _ENTRIES.search(text)
    if match is None:
        raise ValueError("На странице нет рейтинга: адрес категории не тот или разметка изменилась.")
    rows = json_value_at(text, match.end())
    return match.group(1), [
        BenchmarkEntry(
            model_key=str(row.get("modelKey", "")),
            display_name=str(row.get("modelDisplayName", "")),
            organization=str(row.get("modelOrganization", "")),
            score=float(row.get("rating", 0.0)),
            votes=int(row.get("votes", 0)),
        )
        for row in rows
    ]


def fetch(category: ArenaCategory, timeout: float = 60.0) -> list[BenchmarkEntry]:
    """Рейтинг категории с сайта. Ключ рейтинга сверяется с категорией: страница неизвестного
    адреса отдает общий рейтинг, и без проверки категория молча подменилась бы общей."""
    key, rows = parse(get(category.url, timeout))
    if category.expected_key_part not in key:
        raise ValueError(f"Страница {category.url} отдала рейтинг {key}, а не {category.name}.")
    return rows


def fetch_all(categories: Iterable[ArenaCategory] = CATEGORIES, timeout: float = 60.0) -> dict[str, list[BenchmarkEntry]]:
    return {category.key: fetch(category, timeout) for category in categories}
