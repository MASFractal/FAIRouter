from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator

from fai_router import content_review as cr
from fai_router.content_review import ContentReview
from fai_router.enums import ProgrammingLanguage
from fai_router.specifications import Specifications


@dataclass(frozen=True)
class SpecDeviation:
    """Расхождение по одному пункту задания. Отклонение: ноль означает совпадение, единица
    означает полное расхождение. content отличает пункт содержания от пункта формы."""

    field: str
    requested: str
    actual: str
    deviation: float
    content: bool = False

    def __str__(self) -> str:
        """Строка отчета: что заказано и что получено."""
        return f"{self.field}: заказано {self.requested}, получено {self.actual}"


class DiffSpec:
    """Разбор всех расхождений между заданием и фактом по каждому пункту (режим критика).

    Форма сверяется по полям, которые запрос задал явно (Specifications.explicit_fields): угаданное
    моделью «разумное ожидание» требованием не считается. Предмет задачи (область, область науки,
    тип задачи) сверяется всегда. Объем идет одной строкой: символы и слова почти коллинеарны, и две
    строки удваивали его вес. Счетчики сверяются относительным отклонением, шкалы с границами
    (читаемость, доля терминологии, формальность, экспертность) разностью в долях размаха: на них
    относительное отклонение штрафовало ответ лучше заказа сильнее провала. Незаказанные источники
    не штрафуются, блок кода без подписи языка расхождением не считается, язык неизвестен у
    замера, тогда не сверяется. С оценкой содержания разбор включает и его: каждый смысловой пункт
    заказа, каждое ограничение, экспертность ответа против заказанной, каждое проверяемое
    утверждение, расчеты, наполнение структуры, источники и пригодность для дела."""

    # Отклонение, начиная с которого пункт считается проваленным
    MISMATCH_THRESHOLD = cr.MISMATCH_THRESHOLD
    # Оценка, ниже которой критерий, пункт или утверждение считаются проваленными: один порог на
    # критика и на отчет содержания
    PASS_MARK = cr.PASS_MARK

    def __init__(self, deviations: list[SpecDeviation]):
        self.deviations = deviations

    @property
    def total_deviation(self) -> float:
        """Среднее отклонение по всем пунктам, формы и содержания; ноль, если сверять нечего."""
        return _mean([item.deviation for item in self.deviations])

    @property
    def form_deviation(self) -> float:
        """Среднее отклонение по пунктам формы: итог формы равен единице минус это число."""
        return _mean([item.deviation for item in self.deviations if not item.content])

    @property
    def mismatches(self) -> list[SpecDeviation]:
        """Проваленные пункты, худшие первыми."""
        failed = [item for item in self.deviations if item.deviation > self.MISMATCH_THRESHOLD]
        return sorted(failed, key=lambda item: item.deviation, reverse=True)

    @classmethod
    def compare(cls, requested: Specifications, actual: Specifications,
                content: ContentReview | None = None) -> "DiffSpec":
        form = [row for row in _form(requested, actual) if row is not None]
        return cls(form if content is None else form + list(_content(requested, content)))

    def __str__(self) -> str:
        """Отчет критика: список проваленных пунктов."""
        return "\n".join(str(item) for item in self.mismatches)


def _form(requested: Specifications, actual: Specifications) -> list[SpecDeviation | None]:
    def stated(field: str, row: SpecDeviation | None) -> SpecDeviation | None:
        # Строка поля формы, если запрос задал его явно; угаданное не сверяется
        return row if requested.is_explicit(field) else None

    return [
        stated("styleType", _exact("Стиль", requested.style_type, actual.style_type)),
        _volume(requested, actual),
        stated("paragraphCount", _number("Абзацы", requested.paragraph_count, actual.paragraph_count)),
        stated("sectionCount", _number("Разделы", requested.section_count, actual.section_count)),
        stated("listItemCount", _number("Пункты списков", requested.list_item_count, actual.list_item_count)),
        stated("tableCount", _number("Таблицы", requested.table_count, actual.table_count)),
        stated("codeBlockCount", _number("Блоки кода", requested.code_block_count, actual.code_block_count)),
        stated("formulaCount", _number("Формулы", requested.formula_count, actual.formula_count)),
        stated("headingDepth", _number("Глубина заголовков", requested.heading_depth, actual.heading_depth)),
        stated("avgSentenceLength", _number("Средняя длина предложения", requested.avg_sentence_length,
                                            actual.avg_sentence_length)),
        stated("readabilityScore", _scale("Читаемость", requested.readability_score, actual.readability_score, 100)),
        stated("termDensity", _scale("Доля терминологии", requested.term_density, actual.term_density, 1)),
        stated("formalityScore", _scale("Формальность", requested.formality_score, actual.formality_score, 1)),
        stated("language", _language(requested, actual)),
        stated("hasReferences", _references(requested, actual)),
        stated("programmingLanguage", _code(requested, actual)),
        _exact("Область", requested.domain, actual.domain),
        _exact("Область науки", requested.science_field, actual.science_field),
        _exact("Тип задачи", requested.task_kind, actual.task_kind),
    ]


def _content(requested: Specifications, content: ContentReview) -> Iterator[SpecDeviation]:
    """Пункты содержания: у каждого смыслового пункта и ограничения своя строка, у каждого
    проверяемого утверждения тоже. Общая оценка критерия идет строкой, только когда поштучного
    разбора нет: иначе одно расхождение попало бы в разбор дважды."""
    for point in content.points:
        yield SpecDeviation(f"Смысловой пункт «{point.point}»", "раскрыть", _coverage_text(point.coverage),
                            1.0 - point.coverage, True)
    if not content.points:
        yield from _criterion(content, cr.COMPLETENESS)
    for check in content.constraint_checks:
        yield SpecDeviation(f"Ограничение «{check.constraint}»", "соблюсти",
                            "соблюдено" if check.met else "нарушено", 0.0 if check.met else 1.0, True)
    if requested.expert_level > 0 and content.expert_level is not None:
        yield replace(_scale("Экспертность", requested.expert_level, content.expert_level, 1), content=True)
    else:
        yield from _criterion(content, cr.EXPERTISE)
    for claim in content.claims:
        yield SpecDeviation(f"Факт «{claim.text}»", "верно", f"верно с вероятностью {claim.truth:.2f}",
                            1.0 - claim.truth, True)
    for name in (cr.REASONING, cr.STRUCTURE_CONTENT, cr.SOURCE_QUALITY, cr.FIT_FOR_PURPOSE):
        yield from _criterion(content, name)


def _criterion(content: ContentReview, name: str) -> Iterator[SpecDeviation]:
    """Строка критерия содержания; ничего, если критерий к задаче не относится."""
    score = content.get(name)
    if score is not None:
        yield SpecDeviation(name, "1", _text(score), 1.0 - score, True)


def _coverage_text(coverage: float) -> str:
    return "раскрыт" if coverage >= cr.PASS_MARK else "раскрыт частично" if coverage >= 0.3 else "не раскрыт"


def _volume(requested: Specifications, actual: Specifications) -> SpecDeviation | None:
    """Объем одной строкой: та мера, что задана явно. Заданы обе или список неизвестен, тогда
    символы, если они заказаны, иначе слова."""
    symbols, words = requested.is_explicit("symbolLength"), requested.is_explicit("wordLength")
    if symbols and (not words or requested.symbol_length > 0 or requested.word_length == 0):
        return _number("Объем в символах", requested.symbol_length, actual.symbol_length)
    return _number("Объем в словах", requested.word_length, actual.word_length) if words else None


def _language(requested: Specifications, actual: Specifications) -> SpecDeviation | None:
    """Язык сравнивается кодом без региона и регистра; неизвестен с любой стороны, тогда не сверяется."""
    want = Specifications.normalize_language(requested.language)
    got = Specifications.normalize_language(actual.language)
    return None if want is None or got is None else SpecDeviation("Язык", want, got, 0.0 if want == got else 1.0)


def _references(requested: Specifications, actual: Specifications) -> SpecDeviation:
    """Незаказанные источники не штрафуются: штраф за них только при явном запрете, а явность
    известна лишь из списка явных полей."""
    forbidden = requested.explicit_fields is not None
    if requested.has_references == actual.has_references:
        deviation = 0.0
    else:
        deviation = 1.0 if requested.has_references or forbidden else 0.0
    return SpecDeviation("Ссылки на источники", _yes(requested.has_references), _yes(actual.has_references), deviation)


def _code(requested: Specifications, actual: Specifications) -> SpecDeviation | None:
    """Код есть, а языка замер не знает (блоки без подписи): это не расхождение."""
    if actual.programming_language == ProgrammingLanguage.NONE and actual.code_block_count > 0:
        return None
    return _exact("Язык программирования", requested.programming_language, actual.programming_language)


def _number(field: str, requested: float, actual: float) -> SpecDeviation:
    """Счетчик: относительное отклонение от заказанного. Заказан ноль, а получено больше нуля:
    расхождение полное, потому что делить не на что; явный ноль это запрет."""
    if requested == 0:
        deviation = 0.0 if actual == 0 else 1.0
    else:
        deviation = min(1.0, abs(actual - requested) / abs(requested))
    return SpecDeviation(field, _text(requested), _text(actual), deviation)


def _scale(field: str, requested: float, actual: float, size: float) -> SpecDeviation:
    """Шкала с границами: разность в долях размаха, одинаково в обе стороны от заказа."""
    return SpecDeviation(field, _text(requested), _text(actual), min(1.0, abs(actual - requested) / size))


def _exact(field: str, requested: Any, actual: Any) -> SpecDeviation:
    """Категориальный пункт: совпало или нет. Перечисление пишется своим значением, как в C#."""
    return SpecDeviation(field, _name(requested), _name(actual), 0.0 if requested == actual else 1.0)


def _name(value: Any) -> str:
    if value is None:
        return "нет"
    return str(getattr(value, "value", value))


def _yes(value: bool) -> str:
    return "да" if value else "нет"


def _text(value: float) -> str:
    """Число в общем формате без учета локали, четыре значащих знака."""
    return f"{value:.4g}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
