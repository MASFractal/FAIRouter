"""Измерение спецификации текста по факту: все, что считается без обращения к модели."""

from __future__ import annotations

import re

from fai_router.specifications import Specifications

# Гласные русского и английского алфавитов, слог примерно равен гласной.
# Буква ё здесь данные, а не текст: без нее слоги считались бы неверно
VOWELS = set("аеёиоуыэюяaeiouy")

_WORD = re.compile(r"[^\W\d_][\w\-']*|\d+", re.UNICODE)
_HEADING = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+\S", re.MULTILINE)
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s|:-]*-[\s|:-]*\|\s*$", re.MULTILINE)
_CODE_FENCE = re.compile(r"^\s*```", re.MULTILINE)
_FORMULA = re.compile(r"\$\$[^$]+\$\$|\$[^$\n]+\$")
_REFERENCE = re.compile(r"\[[^\]]+\]\([^)]+\)|https?://")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Сокращения, после которых точка не заканчивает предложение. В версии на C# разбиение
# делал SentencesTokenizer из AI.NLP, здесь его заменяет свой короткий список
_ABBREVIATIONS = ("т.е.", "т.д.", "т.п.", "т.к.", "т.н.", "и т.д.", "и т.п.", "см.", "рис.",
                  "табл.", "стр.", "г.", "гг.", "тыс.", "млн", "млрд", "им.", "ул.", "д.", "к.")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-Z0-9«\"(])")
_ABBR_MARK = "\u2024"


def split_sentences(text: str) -> list[str]:
    """Разбиение на предложения с учетом русских сокращений."""
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", _ABBR_MARK))
    parts = _SENTENCE_END.split(protected)
    sentences = [part.replace(_ABBR_MARK, ".").strip() for part in parts]
    return [sentence for sentence in sentences if sentence]


def measure(text: str) -> Specifications:
    """Измеряет спецификацию готового текста. Стиль, доля терминологии и формальность
    остаются по умолчанию, так как они смысловые и измеряются моделью."""
    words = len(_WORD.findall(text))
    sentences = len(split_sentences(text))
    heading_levels = [len(match.group(1)) for match in _HEADING.finditer(text)]

    return Specifications(
        symbol_length=len(text),
        word_length=words,
        paragraph_count=_count_paragraphs(text),
        section_count=heading_levels.count(min(heading_levels)) if heading_levels else 0,
        list_item_count=len(_LIST_ITEM.findall(text)),
        table_count=len(_TABLE_SEPARATOR.findall(text)),
        code_block_count=len(_CODE_FENCE.findall(text)) // 2,
        formula_count=len(_FORMULA.findall(text)),
        heading_depth=max(heading_levels) if heading_levels else 0,
        avg_sentence_length=0.0 if sentences == 0 else words / sentences,
        readability_score=_readability(text, words, sentences),
        language=_detect_language(text),
        has_references=bool(_REFERENCE.search(text)),
    )


def _count_paragraphs(text: str) -> int:
    return sum(1 for block in _PARAGRAPH_BREAK.split(text) if _is_prose(block))


def _is_prose(block: str) -> bool:
    """Абзацем считается сплошной текст: заголовок, список, таблица и код абзацами не
    считаются, иначе замер разойдется с заказом, где под абзацами понимают прозу."""
    first_line = next((line for line in block.split("\n") if line.strip()), None)
    if first_line is None:
        return False
    return not (_HEADING.match(first_line) or _LIST_ITEM.match(first_line)
                or _CODE_FENCE.match(first_line) or first_line.lstrip().startswith("|"))


def _readability(text: str, words: int, sentences: int) -> float:
    """Индекс удобочитаемости Флеша в адаптации Оборневой для русского языка, 0..100."""
    if words == 0 or sentences == 0:
        return 0.0
    syllables = sum(1 for symbol in text.lower() if symbol in VOWELS)
    score = 206.835 - 1.3 * words / sentences - 60.1 * syllables / words
    return min(max(score, 0.0), 100.0)


def _detect_language(text: str) -> str | None:
    """Язык по преобладанию алфавита: различает русский и английский."""
    cyrillic = sum(1 for symbol in text if "а" <= symbol <= "я" or "А" <= symbol <= "Я" or symbol in "ёЁ")
    latin = sum(1 for symbol in text if "a" <= symbol <= "z" or "A" <= symbol <= "Z")
    if cyrillic == 0 and latin == 0:
        return None
    return "ru" if cyrillic >= latin else "en"
