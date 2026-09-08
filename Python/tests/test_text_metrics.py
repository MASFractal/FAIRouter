from fai_router import text_metrics

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
    assert len(spec.feature_vector()) == 23


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
