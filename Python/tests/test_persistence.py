import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from fai_router import FaiRouter
from fai_router.enums import FeedbackType, Style
from fai_router.judge import Judge
from fai_router.persistence import SqliteTraceStore, SqliteWeightsStore
from fai_router.routed_element import RoutedElement
from fai_router.services import InputFeaturesService
from fai_router.settings import RouteWeights, Settings
from fai_router.specifications import Specifications
from fai_router.tracking import Feedback, Tracert

HUMAN = Feedback(FeedbackType.HUMAN, 1.0)


def scalar(path, sql):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(sql).fetchone()[0]
    finally:
        connection.close()


def task():
    return InputFeaturesService.get_features("Разбери задачу и предложи решение.")


def rated(features, name, quality, experience=1000.0):
    element = RoutedElement(name, tps=100, dpmt_inp=1, dpmt_outp=5, ideal_match_vector=features.feature_vector() * quality)
    element.experience = experience
    return element


def trace_of(features, winner, *rivals):
    return Tracert(winner=winner, top_k_elements=[winner, *rivals], input_feature_vector=features.feature_vector(),
                   requested_spec=features.input_specifications)


def test_weights_roundtrip_is_exact(tmp_path):
    store = SqliteWeightsStore(str(tmp_path / "w.db"))
    good = RoutedElement("Подходящая")
    bad = RoutedElement("Неподходящая")
    judge = Judge()
    judge.transformer_w[0, 1] = 0.42
    mean = np.linspace(0, 1, Settings.full_dim())

    store.save([good, bad], judge)
    store.save_task_mean(mean)

    copies = [RoutedElement("Подходящая"), RoutedElement("Неподходящая"), RoutedElement("Невиданная")]
    judge_copy = Judge()
    assert store.load_elements(copies) == 2
    assert store.load_judge(judge_copy)
    assert np.array_equal(copies[0].ideal_match_vector, good.ideal_match_vector)
    assert np.array_equal(judge_copy.transformer_w, judge.transformer_w)
    assert np.array_equal(store.load_task_mean(), mean)
    assert not SqliteWeightsStore(str(tmp_path / "empty.db")).load_judge(Judge())


def test_wrong_dimension_vector_is_skipped_not_fatal(tmp_path):
    """Вектор чужой размерности пропускается, остальные загружаются; среднее при векторе переживает
    сохранение; среднее чужой размерности не применяется."""
    path = str(tmp_path / "w.db")
    features = task()
    good = rated(features, "good", 0.7)
    good.task_mean = np.zeros(Settings.full_dim()) + 0.01
    store = SqliteWeightsStore(path)
    store.save([good])
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("INSERT INTO element_vectors (name, dimension, values_json) VALUES ('bad', 2, '[1,2]')")
    connection.close()
    store.save_task_mean(np.array([1.0, 2.0]))

    fresh_good, fresh_bad = rated(features, "good", 0.1), rated(features, "bad", 0.1)
    bad_before = fresh_bad.ideal_match_vector.copy()
    assert store.load_vectors([fresh_good, fresh_bad]) == ["good"]
    assert np.array_equal(fresh_good.ideal_match_vector, good.ideal_match_vector)
    assert np.array_equal(fresh_good.task_mean, good.task_mean)
    assert np.array_equal(fresh_bad.ideal_match_vector, bad_before)
    assert store.load_task_mean() is None


def test_old_database_is_migrated_in_place(tmp_path):
    """База прежней версии доводится до новой схемы без потери ходов; журнал в режиме WAL; вектор
    без своего среднего получает общее среднее из таблицы task_mean."""
    path = str(tmp_path / "old.db")
    features = task()
    legacy_mean = np.zeros(Settings.full_dim()) + 0.05
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript(f"""
            CREATE TABLE rounds (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, prompt TEXT,
                winner TEXT NOT NULL, rivals_json TEXT NOT NULL, features_json TEXT NOT NULL, score REAL NOT NULL DEFAULT 0,
                is_exploration INTEGER NOT NULL DEFAULT 0, requested_json TEXT, actual_json TEXT, feedback_type INTEGER,
                feedback_score REAL);
            CREATE TABLE element_vectors (name TEXT PRIMARY KEY, dimension INTEGER NOT NULL, values_json TEXT NOT NULL);
            CREATE TABLE task_mean (id INTEGER PRIMARY KEY CHECK (id = 1), values_json TEXT NOT NULL);
            INSERT INTO rounds (created_at, winner, rivals_json, features_json, feedback_type, feedback_score)
            VALUES ('2026-01-01T00:00:00.0000000Z', 'a', '["b"]', '[1,0]', 1, 1.0);
            INSERT INTO element_vectors VALUES ('a', {Settings.full_dim()}, '{json.dumps([0.1] * Settings.full_dim())}');
            INSERT INTO task_mean VALUES (1, '{json.dumps(legacy_mean.tolist())}');
        """)
    connection.close()

    store = SqliteTraceStore(path)
    a, b = rated(features, "a", 0.5), rated(features, "b", 0.5)
    rounds = store.read_rated([a, b])
    assert len(rounds) == 1 and rounds[0].trace.winner is a and rounds[0].trace.forecast is None
    assert scalar(path, "PRAGMA user_version") == 2
    assert scalar(path, "PRAGMA journal_mode") == "wal"
    assert scalar(path, "SELECT COUNT(*) FROM pragma_table_info('element_vectors') WHERE name = 'mean_json'") == 1

    assert SqliteWeightsStore(path).load_vectors([a]) == ["a"]
    assert np.array_equal(a.task_mean, legacy_mean)


def test_rated_rounds_come_in_order_and_once(tmp_path):
    """Обучение идет по порядку записи, только по необученным, а снятый соперник выпадает из пары,
    а не из выборки. Повтор кандидата в каталоге журнал не роняет."""
    store = SqliteTraceStore(str(tmp_path / "t.db"))
    features = task()
    a, b, gone = rated(features, "a", 0.5), rated(features, "b", 0.5), rated(features, "снятая", 0.5)
    first = store.append(trace_of(features, a, b, gone))
    second = store.append(trace_of(features, b, a))
    store.set_feedback(first, HUMAN)
    store.set_feedback(second, HUMAN)

    rounds = store.read_rated([a, b, a])
    assert [item.id for item in rounds] == [first, second]
    assert [element.name for element in rounds[0].trace.top_k_elements] == ["a", "b"]

    store.mark_trained([first])
    assert [item.id for item in store.read_rated([a, b], untrained_only=True)] == [second]
    store.set_feedback(first, Feedback(FeedbackType.HUMAN, 0.0))
    assert len(store.read_rated([a, b], untrained_only=True)) == 2


def test_statistics_count_auto_feedback_and_shrink_variance(tmp_path):
    """Автоотзывы входят в опыт с весом AUTO_FEEDBACK_WEIGHT; два одинаковых отзыва не дают
    нулевой дисперсии: она стянута к неизвестной."""
    store = SqliteTraceStore(str(tmp_path / "t.db"))
    features = task()
    a, b = rated(features, "a", 0.5, 0), rated(features, "b", 0.5, 0)
    store.set_feedback(store.append(trace_of(features, a, b)), HUMAN)
    for _ in range(4):
        store.set_feedback(store.append(trace_of(features, a, b)), Feedback(FeedbackType.AUTO, 0.8))
    store.set_feedback(store.append(trace_of(features, b, a)), HUMAN)
    store.set_feedback(store.append(trace_of(features, b, a)), HUMAN)

    assert store.load_statistics([a, b, b]) == 2
    assert a.experience == pytest.approx(1 + 4 * Settings.AUTO_FEEDBACK_WEIGHT)
    assert b.experience == pytest.approx(2)
    # Два одинаковых отзыва: разброс ноль, но дисперсия стянута к неизвестной 0,25 с силой 1
    assert b.score_variance == pytest.approx(Settings.UNKNOWN_VARIANCE / 2)
    # В обучающую выборку автоотзывы входят: роутер по ним учится, просто слабее
    assert len(store.read_rated([a, b])) == 7


def test_calibration_uses_forecast_at_selection_time(tmp_path):
    """Калибровка берет прогноз в момент выбора и только человеческие отзывы."""
    store = SqliteTraceStore(str(tmp_path / "t.db"))
    features = task()
    a, b = rated(features, "a", 0.5), rated(features, "b", 0.5)
    human = trace_of(features, a, b)
    human.forecast = 0.123
    store.set_feedback(store.append(human), HUMAN)
    auto = trace_of(features, a, b)
    auto.forecast = 0.9
    store.set_feedback(store.append(auto), Feedback(FeedbackType.AUTO, 0.2))
    a.ideal_match_vector = features.feature_vector() * 0.99
    assert store.read_calibration([a, b]) == [(0.123, 1.0)]


def test_feature_mean_reads_only_recent_rounds_and_purge_forgets_old_ones(tmp_path):
    store = SqliteTraceStore(str(tmp_path / "t.db"))
    features = task()
    a = rated(features, "a", 0.5)
    for _ in range(3):
        store.append(trace_of(features, a), prompt="текст")

    assert store.get_feature_mean(limit=2) is not None
    now = datetime.now(timezone.utc)
    assert store.purge(now + timedelta(minutes=1), prompts_only=True) == 3
    assert store.purge(now - timedelta(days=1)) == 0
    assert store.purge(now + timedelta(minutes=1)) == 3
    assert store.count() == (0, 0)


def test_trace_journal_accumulates_and_restores(tmp_path):
    traces = SqliteTraceStore(str(tmp_path / "t.db"))
    writer, coder = RoutedElement("Писатель"), RoutedElement("Кодер")
    science = Specifications(style_type=Style.SCIENTIFIC, symbol_length=4000, term_density=0.7,
                             explicit_fields=["symbolLength"])
    features = InputFeaturesService.get_features("задача")
    features.input_specifications = science
    trace = Tracert(winner=writer, top_k_elements=[writer, coder],
                    input_feature_vector=features.feature_vector(), is_exploration=True, failed=["Кодер"])

    rated_id = traces.append(trace, science, science, "научный обзор")
    traces.append(trace, science, None, "без отзыва")
    assert traces.count() == (2, 0)

    traces.set_feedback(rated_id, Feedback(FeedbackType.HUMAN, 0.9))
    assert traces.count() == (2, 1)
    with pytest.raises(ValueError):
        traces.set_feedback(999, Feedback())

    sample = traces.read_rated([writer, coder])
    assert len(sample) == 1
    assert sample[0].trace.winner is writer and sample[0].trace.is_exploration
    assert sample[0].requested.style_type == Style.SCIENTIFIC and sample[0].actual is not None
    assert sample[0].requested.explicit_fields == ["symbolLength"]
    assert len(sample[0].trace.input_feature_vector) == Settings.full_dim()
    assert traces.read_rated([coder]) == []

    assert traces.load_statistics([writer, coder]) == 1
    assert writer.experience == 1
    assert np.allclose(traces.get_feature_mean(), features.feature_vector())


def test_router_trains_once_saves_only_trained_and_keeps_global_mean(tmp_path):
    """Роутер учит каждый ход один раз, сохраняет только обученные векторы и не трогает общее
    среднее задач: прежде load и первый train задавали Settings.task_mean, и прогноз необученных
    кандидатов пересчитывался в чужом пространстве."""
    path = str(tmp_path / "r.db")
    features = task()
    SqliteWeightsStore(path).save_task_mean(np.zeros(Settings.full_dim()) + 0.05)

    def candidates():
        return [rated(features, "a", 0.5), rated(features, "b", 0.6), rated(features, "c", 0.4)]

    first = candidates()
    untrained = first[2].get_quality_score(features.feature_vector())
    router = FaiRouter(first, lambda c, m: "ответ", path, measure=False)
    assert first[2].get_quality_score(features.feature_vector()) == pytest.approx(untrained, abs=1e-12)
    assert Settings.task_mean is None

    round_id = router.traces.append(trace_of(features, first[0], first[1]))
    router.feedback(round_id, 1.0)
    loss = router.train()
    assert loss.rounds == 1 and loss.router > 0 and loss.judge == 0 and float(loss) == loss.router
    assert router.train().rounds == 0
    assert Settings.task_mean is None
    assert router.unsaved
    router.save()
    assert not router.unsaved

    second = candidates()
    FaiRouter(second, lambda c, m: "ответ", path, measure=False)
    assert np.array_equal(first[0].ideal_match_vector, second[0].ideal_match_vector)
    assert np.array_equal(candidates()[1].ideal_match_vector, second[1].ideal_match_vector)
    assert scalar(path, "SELECT COUNT(*) FROM element_vectors") == 1
    assert FaiRouter(candidates(), lambda c, m: "ответ", path, measure=False).train().rounds == 0


def test_explicit_temperature_scale_is_kept():
    """Явный множитель температуры берется как есть, даже совпадая с общим; профили получают
    множитель роутера."""
    router = FaiRouter([rated(task(), "a", 0.5)], lambda c, m: "ответ", measure=False, temperature_scale=10)
    assert router.weights_for(RouteWeights.quality()).temperature_scale == 10
    assert router.weights_for(RouteWeights(0.5, 0.25, 0.25, Settings.temperature_scale)).temperature_scale == \
        Settings.temperature_scale
    assert router.weights_for("price").temperature_scale == 10


def test_connections_are_closed(tmp_path):
    """Хранилище не держит файл открытым: после действий базу можно удалить (на Windows открытое
    соединение мешало бы)."""
    path = tmp_path / "t.db"
    store = SqliteTraceStore(str(path))
    features = task()
    store.set_feedback(store.append(trace_of(features, rated(features, "a", 0.5))), HUMAN)
    store.count()
    SqliteWeightsStore(str(path)).save_judge(Judge())
    for suffix in ("", "-wal", "-shm"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists():
            candidate.unlink()
    assert not path.exists()
