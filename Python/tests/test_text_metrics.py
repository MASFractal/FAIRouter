from fai_router import text_metrics
from fai_router.enums import ProgrammingLanguage

ANSWER = """# Отчет по проекту

Первый абзац вводной части. Здесь две фразы, т.е. проверяем сокращения.

## Раздел данных

- первый пункт
- второй пункт
- третий пункт

| Колонка | Значение |
|---------|----------|
| Альфа   | 10       |

Формула стоимости $E = mc^2$ приведена для примера.

```csharp
var x = 1;
```

## Раздел выводов

Подробности в [источнике](https://example.com/doc).
"""


def test_structure_is_measured_like_csharp():
    spec = text_metrics.measure(ANSWER)
    assert spec.section_count == 1
    assert spec.list_item_count == 3
    assert spec.table_count == 1
    assert spec.code_block_count == 1
    assert spec.formula_count == 1
    assert spec.heading_depth == 2
    assert spec.paragraph_count == 3
    assert spec.language == "ru"
    assert spec.has_references
    assert spec.programming_language == ProgrammingLanguage.CSHARP
    assert len(spec.feature_vector()) == 104


def test_abbreviations_do_not_split_sentences():
    assert len(text_metrics.split_sentences("Взяли данные, т.е. таблицу. Потом посчитали.")) == 2


def test_yo_counts_as_vowel_and_cyrillic():
    with_yo = text_metrics.measure("Ёж нёс ёлку через тёмный лес. Он вёз её сестрёнке в тёплый дом.")
    without = text_metrics.measure("Еж нес елку через темный лес. Он вез ее сестренке в теплый дом.")
    assert with_yo.language == "ru"
    assert abs(with_yo.readability_score - without.readability_score) < 1e-9


def test_empty_text_is_safe():
    spec = text_metrics.measure("")
    assert spec.word_length == 0 and spec.readability_score == 0 and spec.language is None


# Те же фразы и числа, что в C#-тестах замера (MeasureTests.cs): обе версии меряют одинаково


def test_compound_words_count_once():
    assert text_metrics.count_words("Рост на 15% в 2025-м году, модель gpt-4o и 3D-печать.") == 10
    assert text_metrics.count_words("snake_case") == 1


def test_sentences_respect_abbreviations_only_at_word_start():
    assert text_metrics.count_sentences("Прошел год. Новый начался.") == 2
    assert text_metrics.count_sentences("Это видно на рис. 3 и в табл. 2.") == 1
    assert text_metrics.count_sentences("Заголовок без точки\nстрока списка") == 2


def test_formulas_need_tex_markup():
    assert text_metrics.measure("Цена $5 и $10 за штуку.").formula_count == 0
    assert text_metrics.measure(r"Пусть $x^2$ и $$a+b$$, а также \(\alpha\).").formula_count == 3


def test_code_is_not_prose():
    spec = text_metrics.measure("Пример ниже.\n\n```python\n# comment\nprint('hello world')\n```\n")
    assert spec.section_count == 0
    assert spec.language == "ru"
    assert spec.code_block_count == 1
    assert spec.programming_language == ProgrammingLanguage.PYTHON


def test_unknown_language_is_not_english():
    assert text_metrics.detect_language("The quick brown fox jumps over the lazy dog.") == "en"
    assert text_metrics.detect_language("Le café est très agréable près de la gare, déjà ouvert à midi.") is None
    assert text_metrics.detect_language("Він прийшов і її побачив.") is None
    assert text_metrics.detect_language("你好，世界") is None


def test_english_readability_uses_reading_ease():
    text = ("The committee reviewed the proposal carefully before the meeting. "
            "Several members raised concerns about the budget and the timeline. "
            "After a long discussion, they agreed to postpone the final decision until next month.")
    assert 34 <= text_metrics.measure(text).readability_score <= 54


def test_unlabeled_code_gives_no_language_and_ties_go_to_the_first():
    assert text_metrics.measure("Код:\n\n```\nprint(1)\n```\n").programming_language == ProgrammingLanguage.NONE
    mixed = "```go\nx\n```\n\n```python\ny\n```\n"
    assert text_metrics.programming_language_of(mixed) == ProgrammingLanguage.GO
    assert text_metrics.programming_language_of("```brainfuck\n+\n```") == ProgrammingLanguage.OTHER


def test_references_cover_doi_and_bibliography():
    assert text_metrics.measure("См. doi: 10.1000/182 в конце.").has_references
    assert text_metrics.measure("Текст.\n\n[1] Иванов И. Книга. 2020.").has_references
    assert text_metrics.measure("Текст.\n\n[^1]: Сноска.").has_references
    assert not text_metrics.measure("Просто текст без источников.").has_references
