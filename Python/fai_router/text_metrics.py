"""Измерение спецификации текста по факту: все, что считается без обращения к модели.

Правила общие с версией на C# (TextMetrics) и описаны здесь буквально, чтобы обе стороны мерили
одинаково.

* Слово: буквы и цифры, внутри соединенные дефисом, апострофом или подчеркиванием (2025-м, gpt-4o,
  3D-печать, snake_case это по одному слову, «15%» это слово 15).
* Код (ограды ``` и `встроенный`) в прозу не входит: структура, формулы, источники, предложения,
  читаемость и язык меряются без него. Объем в символах и словах меряется по всему тексту.
* Предложение: строка прозы делится после . ! ? … перед пробелом и заглавной буквой, цифрой,
  кавычкой или скобкой; конец строки тоже граница. Точки в сокращениях (т.е., г., рис. и др., только
  с начала слова) границей не считаются.
* Читаемость: индекс Флеша по прозе. Для русского коэффициенты Оборневой 206,835 - 1,3 ASL - 60,1 ASW,
  для остальных Flesch Reading Ease 206,835 - 1,015 ASL - 84,6 ASW; слог это русская гласная или
  группа латинских гласных подряд, немое окончание слогом не считается, на слово не меньше слога.
* Язык: ru или en по преобладающему алфавиту прозы; буквы вне русского алфавита у кириллицы или с
  диакритикой у латиницы (больше 0,5 %), иная письменность или пустая проза значат «неизвестно».
* Формула: $$…$$, \\[…\\], \\(…\\) и $…$ только с признаками TeX внутри (\\команда, ^, _, {):
  «$5 и $10» формулой не считается.
"""

from __future__ import annotations

import re
from collections import Counter

from fai_router.enums import ProgrammingLanguage
from fai_router.specifications import Specifications

# Доля букв вне основного алфавита, после которой язык считается неизвестным
FOREIGN_LETTER_SHARE = 0.005

# Русские гласные, слог примерно равен гласной. Буква ё здесь данные, а не текст: без нее слоги
# считались бы неверно
RUSSIAN_VOWELS = "аеёиоуыэюя"

_ABBREVIATION_DOT = "․"

_WORD = re.compile(r"[^\W_]+(?:['’\-_][^\W_]+)*")
_HEADING = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+\S", re.MULTILINE)
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s|:-]*-[\s|:-]*\|\s*$", re.MULTILINE)
_CODE_FENCE = re.compile(r"^\s*```", re.MULTILINE)
_CODE_FENCE_INFO = re.compile(r"^\s*```[ \t]*([A-Za-z0-9#+.\-]+)", re.MULTILINE)
_CODE_BLOCK = re.compile(r"^[ \t]*```[^\n]*\n[\s\S]*?^[ \t]*```[ \t]*$", re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_LINK_TARGET = re.compile(r"!?\[([^\]]*)\]\([^)]*\)|https?://\S+")
_LINE_MARKER = re.compile(r"^\s*(#{1,6}\s+|[-*+>]\s+|\d+[.)]\s+)", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9«\"“(\[])")
_LATIN_VOWELS = re.compile(r"[aeiouy]+", re.IGNORECASE)
_SILENT_ENDING = re.compile(r"[^aeiouyl]e$|[^aeiouytd]ed$|[^aeiouysxzcg]es$", re.IGNORECASE)
_FORMULA = re.compile(r"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?=[^$\n]*[\\^_{])[^$\n]+\$")
# Источник: ссылка разметки, адрес, DOI, строка списка литературы [1] … или сноска [^1]: …
_REFERENCE = re.compile(
    r"\[[^\]]+\]\([^)]+\)|https?://|\bdoi:\s*10\.\d{4,9}/|\b10\.\d{4,9}/\S+|^\s*\[\d{1,3}\]\s+\S|^\[\^[^\]]+\]:",
    re.MULTILINE | re.IGNORECASE)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Сокращения, после которых точка не заканчивает предложение. Совпадение только с начала слова:
# иначе «год.» прятал бы конец предложения как «д.»
_ABBREVIATION = re.compile(
    r"(?<![^\W_])(?:и т\.д\.|и т\.п\.|т\.е\.|т\.д\.|т\.п\.|т\.к\.|т\.н\.|см\.|рис\.|табл\.|стр\.|гг\.|г\.|тыс\.|"
    r"млн\.|млрд\.|им\.|ул\.|д\.|к\.|e\.g\.|i\.e\.|mr\.|mrs\.|ms\.|dr\.|vs\.)",
    re.IGNORECASE)

# Подписи ограждений кода по языкам; подпись вне таблицы означает язык вне списка
_FENCE_LANGUAGES = {
    "python": ProgrammingLanguage.PYTHON, "py": ProgrammingLanguage.PYTHON,
    "javascript": ProgrammingLanguage.JAVASCRIPT, "js": ProgrammingLanguage.JAVASCRIPT,
    "jsx": ProgrammingLanguage.JAVASCRIPT, "node": ProgrammingLanguage.JAVASCRIPT,
    "typescript": ProgrammingLanguage.TYPESCRIPT, "ts": ProgrammingLanguage.TYPESCRIPT,
    "tsx": ProgrammingLanguage.TYPESCRIPT,
    "csharp": ProgrammingLanguage.CSHARP, "cs": ProgrammingLanguage.CSHARP, "c#": ProgrammingLanguage.CSHARP,
    "java": ProgrammingLanguage.JAVA,
    "go": ProgrammingLanguage.GO, "golang": ProgrammingLanguage.GO,
    "rust": ProgrammingLanguage.RUST, "rs": ProgrammingLanguage.RUST,
    "cpp": ProgrammingLanguage.CPP, "c++": ProgrammingLanguage.CPP, "cc": ProgrammingLanguage.CPP,
    "cxx": ProgrammingLanguage.CPP, "c": ProgrammingLanguage.CPP, "h": ProgrammingLanguage.CPP,
    "sql": ProgrammingLanguage.SQL, "postgresql": ProgrammingLanguage.SQL,
    "mysql": ProgrammingLanguage.SQL, "psql": ProgrammingLanguage.SQL,
    "html": ProgrammingLanguage.HTML, "xml": ProgrammingLanguage.HTML,
    "css": ProgrammingLanguage.HTML, "svg": ProgrammingLanguage.HTML,
}


def measure(text: str) -> Specifications:
    """Измеряет спецификацию готового текста. Стиль, доля терминологии и формальность
    остаются по умолчанию, так как они смысловые и измеряются моделью."""
    body = _without_code(text)
    prose = _prose(body)
    language = detect_language(prose)
    prose_words = _WORD.findall(prose)
    sentences = count_sentences(prose)
    heading_levels = [len(match.group(1)) for match in _HEADING.finditer(body)]

    return Specifications(
        symbol_length=len(text),
        word_length=count_words(text),
        paragraph_count=_count_paragraphs(body),
        section_count=heading_levels.count(min(heading_levels)) if heading_levels else 0,
        list_item_count=len(_LIST_ITEM.findall(body)),
        table_count=len(_TABLE_SEPARATOR.findall(body)),
        code_block_count=len(_CODE_FENCE.findall(text)) // 2,
        formula_count=len(_FORMULA.findall(body)),
        heading_depth=max(heading_levels) if heading_levels else 0,
        avg_sentence_length=0.0 if sentences == 0 else len(prose_words) / sentences,
        readability_score=_readability(prose_words, sentences, language),
        language=language,
        has_references=bool(_REFERENCE.search(body)),
        programming_language=programming_language_of(text),
    )


def count_words(text: str) -> int:
    """Число слов по общему правилу (см. описание модуля)."""
    return len(_WORD.findall(text))


def count_sentences(text: str) -> int:
    """Число предложений по общему правилу (см. описание модуля); text это проза без кода."""
    return sum(1 for line in text.split("\n")
               for part in _SENTENCE_END.split(_protect_abbreviations(line))
               if any(symbol.isalpha() or symbol.isdecimal() for symbol in part))


def split_sentences(text: str) -> list[str]:
    """Предложения по тому же правилу, что count_sentences, с восстановленными точками сокращений."""
    return [part.replace(_ABBREVIATION_DOT, ".").strip()
            for line in text.split("\n")
            for part in _SENTENCE_END.split(_protect_abbreviations(line))
            if any(symbol.isalpha() or symbol.isdecimal() for symbol in part)]


def detect_language(prose: str) -> str | None:
    """Язык прозы кодом ISO 639-1: ru или en; None, если язык иной или не определяется."""
    letters = cyrillic = russian = latin = ascii_letters = 0
    for symbol in prose:
        if not symbol.isalpha():
            continue
        letters += 1
        if "Ѐ" <= symbol <= "ӿ":
            cyrillic += 1
            russian += "а" <= symbol <= "я" or "А" <= symbol <= "Я" or symbol in "ёЁ"
        elif "a" <= symbol <= "z" or "A" <= symbol <= "Z" or "À" <= symbol <= "ɏ":
            latin += 1
            ascii_letters += symbol < "\u0080"

    if cyrillic >= latin and cyrillic * 2 > letters:
        return None if cyrillic - russian > FOREIGN_LETTER_SHARE * cyrillic else "ru"
    if latin > cyrillic and latin * 2 > letters:
        return None if latin - ascii_letters > FOREIGN_LETTER_SHARE * latin else "en"
    return None


def programming_language_of(text: str) -> ProgrammingLanguage:
    """Язык кода в тексте по подписям ограждений: самый частый из подписанных, при равенстве тот,
    что встретился раньше. Подпись вне таблицы дает язык вне списка (OTHER). Блоков нет, тогда NONE.
    Блоки есть, а подписей нет, тогда язык неизвестен и это тоже NONE (вместе с ненулевым числом
    блоков): прежде это был OTHER, и заказанный Python с неподписанным блоком считался расхождением."""
    tagged = [_FENCE_LANGUAGES.get(tag.lower(), ProgrammingLanguage.OTHER) for tag in _CODE_FENCE_INFO.findall(text)]
    # Counter помнит порядок первого появления, а most_common при равенстве его сохраняет
    return Counter(tagged).most_common(1)[0][0] if tagged else ProgrammingLanguage.NONE


def _protect_abbreviations(line: str) -> str:
    return _ABBREVIATION.sub(lambda match: match.group(0).replace(".", _ABBREVIATION_DOT), line)


def _without_code(text: str) -> str:
    """Текст без блоков кода: блок заменяется переводом строки, и соседние абзацы не склеиваются."""
    return _CODE_BLOCK.sub("\n", text)


def _prose(body: str) -> str:
    """Проза: без встроенного кода, адресов и разметки ссылок (текст ссылки остается), без меток
    заголовков, списков и цитат, без строк-разделителей таблиц; ячейки таблиц идут отдельными строками."""
    text = _TABLE_SEPARATOR.sub("", body)
    text = _INLINE_CODE.sub(" ", text)
    text = _LINK_TARGET.sub(lambda match: match.group(1) or "", text)
    text = _LINE_MARKER.sub("", text)
    return text.replace("|", "\n").replace("*", "")


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


def _readability(words: list[str], sentences: int, language: str | None) -> float:
    """Индекс удобочитаемости Флеша, 0..100: коэффициенты Оборневой для русского, Reading Ease для прочих."""
    if not words or sentences == 0:
        return 0.0
    asl = len(words) / sentences
    asw = sum(_syllables(word) for word in words) / len(words)
    score = (206.835 - 1.3 * asl - 60.1 * asw if language == "ru"
             else 206.835 - 1.015 * asl - 84.6 * asw)
    return min(max(score, 0.0), 100.0)


def _syllables(word: str) -> int:
    """Слоги слова: русские гласные и группы латинских гласных подряд без немого окончания, не меньше одного."""
    latin = len(_LATIN_VOWELS.findall(word))
    if latin > 1 and _SILENT_ENDING.search(word):
        latin -= 1
    return max(1, sum(1 for symbol in word.lower() if symbol in RUSSIAN_VOWELS) + latin)
