"""Начальный вектор кандидата из внешних замеров качества моделей.

У замеров есть оценки моделей по сериям (предпочтения людей, отраслевые индексы, фактология по
областям), а у роутера есть механизм начальных весов из замеров по типам задач
(from_measurements). Здесь серии превращаются в типы задач: каждой серии соответствует профиль
признаков, а оценка модели в ней становится качеством на этом профиле.

Профили лежат в общем файле data/benchmark_profiles.json: версия на C# читает тот же файл, и
расходиться профилям негде. Скорость и доля рассуждений в цене задачи дают начальные скорость и
поправку цены кандидата, пока своих замеров нет."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fai_router.settings import GLOBAL_MEAN, Settings
from fai_router.specifications import Specifications
from fai_router.tracking import InputFeatures
from fai_router.training.quality_prior import fit

if TYPE_CHECKING:
    from fai_router.benchmarks import BenchmarkSnapshot

PROFILES_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_profiles.json"

# Поправка цены из доли рассуждений: во сколько раз задача дороже прайса ответа. Верх нужен
# потому, что доля рассуждений бывает под 0,99, а один такой замер не повод считать модель
# стократно дороже
MAX_COST_RATIO = 5.0

# Серии с долей рассуждений в цене задачи: по ним считается начальная поправка цены
_REASONING_SHARE = "bench:reasoning-share/"

# Серия скорости, токенов в секунду
_SPEED = "bench:speed"


def _load() -> dict[str, Any]:
    with open(PROFILES_PATH, encoding="utf-8") as file:
        return json.load(file)


_SOURCE = _load()


def task(item: dict[str, Any]) -> InputFeatures:
    """Задача по записи профиля: base.spec, затем шаблон, затем spec профиля."""
    base = _SOURCE["base"]
    spec = dict(base["spec"])
    if "template" in item:
        spec.update(_SOURCE["templates"][item["template"]])
    spec.update(item.get("spec", {}))
    return InputFeatures(
        input_len=float(item.get("input_len", base["input_len"])),
        len_answer=float(item.get("len_answer", base["len_answer"])),
        input_specifications=Specifications.from_dict(spec),
        turn_count=int(item.get("turn_count", base["turn_count"])),
    )


def typical_task() -> InputFeatures:
    """Опорная задача: обычный развернутый ответ пользователю."""
    return task({})


# Профили по ключам серий, в порядке файла
PROFILES: dict[str, list[InputFeatures]] = {
    key: [task(item) for item in items] for key, items in _SOURCE["profiles"].items()
}


def measurements(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> list[tuple[np.ndarray, float]]:
    """Пары «вектор задачи, качество» по всем сериям, где модель есть."""
    return [(task, quality) for task, quality, _ in _points(snapshot, openrouter_id)]


def vector(snapshot: "BenchmarkSnapshot", openrouter_id: str,
           mean: "np.ndarray | None | object" = GLOBAL_MEAN) -> np.ndarray | None:
    """Начальный вектор кандидата по рейтингам в пространстве среднего mean (не задано, значит
    общее Settings.task_mean; None, значит без вычитания); None, если модели нет ни в одной серии.

    Точка серии весит 1/(число профилей серии): серия весит одинаково, сколько бы профилей задач за
    ней ни стояло. Прежде серия с четырьмя профилями тянула прогноз вчетверо сильнее серии с одним."""
    points = _points(snapshot, openrouter_id)
    if not points:
        return None
    return fit([(task, quality) for task, quality, _ in points], [weight for _, _, weight in points], mean)


def known_series(snapshot: "BenchmarkSnapshot") -> int:
    """Сколько серий снимка известно профилям. Ноль означает, что снимок собран под другие имена
    серий: приора по нему не получит ни одна модель, и это тише пустого снимка."""
    return sum(1 for key in PROFILES if key in snapshot.entries)


def field_quality(snapshot: "BenchmarkSnapshot") -> float | None:
    """Уровень поля: средняя доля качества по всем строкам серий, известных профилям. С него
    стартует модель, которой в рейтингах нет (uniform()); None, если известных серий нет.

    Качество в серии это доля между худшим и лучшим, и даже у сильнейших моделей в среднем по
    сериям оно около половины: лидер в каждой серии свой. Балл возможностей каталога лежит на другой
    шкале, около 0.8 у любой современной модели, и подставленный в uniform() напрямую он ставил
    незнакомую модель выше всех оцененных на любой задаче. Если снимок несет среднюю полной серии
    (bounds), берется она: строки обрезанного снимка это верх серии, и средняя по ним завышала бы
    уровень поля."""
    series = [share for share in (snapshot.mean_share(key) for key in PROFILES) if share is not None]
    rows = sum(count for _, count in series)
    return None if rows == 0 else sum(share * count for share, count in series) / rows


# Вектор uniform для единицы и среднее, в котором он построен: подгонка линейна по качеству, а
# безрейтинговых моделей в каталоге сотни, и подгонка на каждую задерживала бы первый ход
_uniform_unit: tuple[object, np.ndarray] | None = None


def uniform(quality: float, mean: "np.ndarray | None | object" = GLOBAL_MEAN) -> np.ndarray:
    """Начальный вектор модели без рейтингов: качество одинаково во всех сериях, в пространстве
    среднего mean (как у vector()).

    Та же подгонка по тем же профилям и с теми же весами точек, что у vector(), поэтому шкала общая
    с моделями из рейтингов: вектор по одной опорной задаче сжимается иначе, и безрейтинговая модель
    обгоняла рейтинговые. Качество задается на шкале серий: незнакомой модели подходит уровень поля
    (field_quality()), а балл каталога годится лишь как множитель к нему, не вместо него."""
    global _uniform_unit
    mean = Settings.task_mean if mean is GLOBAL_MEAN else mean
    unit = _uniform_unit
    if unit is None or unit[0] is not mean:
        points = [(item.feature_vector(), 1.0, 1.0 / len(tasks)) for tasks in PROFILES.values() for item in tasks]
        unit = _uniform_unit = (mean, fit([(task, value) for task, value, _ in points],
                                          [weight for _, _, weight in points], mean))
    return unit[1] * quality


def tokens_per_second(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> float | None:
    """Скорость модели по внешнему замеру, токенов в секунду; None, если замера нет."""
    speed = snapshot.value(_SPEED, openrouter_id)
    return speed if speed is not None and speed > 0 else None


def default_tokens_per_second(snapshot: "BenchmarkSnapshot") -> float | None:
    """Скорость модели, которой в серии скорости нет: нижняя граница серии. Серия во встроенном
    снимке обрезана до самых быстрых, и модель вне нее медленнее последней строки, но не в разы:
    прежнее умолчание 50 токенов в секунду делало модель на 160 втрое медленнее модели на 170.
    Серии нет, значит None."""
    bounds = snapshot.bounds_of(_SPEED)
    return bounds.low if bounds is not None and bounds.low > 0 else None


def cost_ratio(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> float | None:
    """Начальная поправка цены: задача обходится дороже прайса ответа на долю рассуждений.
    Среднее по сериям, где модель есть; None, если ее нет ни в одной."""
    shares = [share for share in (snapshot.value(key, openrouter_id) for key in snapshot.entries
                                  if key.startswith(_REASONING_SHARE))
              if share is not None]
    if not shares:
        return None
    share = sum(shares) / len(shares)
    return min(max(1.0 / max(1.0 - share, 1e-9), 1.0), MAX_COST_RATIO)


def _points(snapshot: "BenchmarkSnapshot", openrouter_id: str) -> list[tuple[np.ndarray, float, float]]:
    """Точки подгонки: задача профиля, качество модели в серии и вес 1/(число профилей серии)."""
    points = []
    for key, tasks in PROFILES.items():
        quality = snapshot.quality(key, openrouter_id)
        if quality is not None:
            points.extend((item.feature_vector(), quality, 1.0 / len(tasks)) for item in tasks)
    return points
