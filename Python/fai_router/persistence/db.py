"""Открытие базы и ее схема: веса и накопленный опыт лежат в одном файле, поэтому и путь к нему
превращается в соединение одинаково, и схема у обоих хранилищ одна. Схема общая с версией на C#
(SqliteDb).

Версия схемы хранится в PRAGMA user_version. База прежней версии доводится до текущей добавлением
колонок, а не пересозданием: журнал и веса в ней остаются. Журнал пишется в режиме WAL, чтобы запись
хода не ждала чтения обучения и наоборот."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

# Версия схемы. 1: таблицы журнала, весов, среднего и судьи. 2: прогноз в момент выбора, отметка
# обученного хода и отказавшие кандидаты в журнале, среднее задач у каждого вектора
SCHEMA_VERSION = 2

BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT    NOT NULL,
    prompt         TEXT,
    winner         TEXT    NOT NULL,
    rivals_json    TEXT    NOT NULL,
    features_json  TEXT    NOT NULL,
    score          REAL    NOT NULL DEFAULT 0,
    is_exploration INTEGER NOT NULL DEFAULT 0,
    requested_json TEXT,
    actual_json    TEXT,
    feedback_type  INTEGER,
    feedback_score REAL
);
CREATE TABLE IF NOT EXISTS element_vectors (
    name        TEXT    PRIMARY KEY,
    dimension   INTEGER NOT NULL,
    values_json TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS task_mean (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    values_json TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS judge_matrix (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    rows_count    INTEGER NOT NULL,
    columns_count INTEGER NOT NULL,
    values_json   TEXT    NOT NULL
);
"""

# Колонки второй версии: таблица, имя, определение
_COLUMNS = (
    ("rounds", "predicted_quality", "REAL"),
    ("rounds", "trained", "INTEGER NOT NULL DEFAULT 0"),
    ("rounds", "failed_json", "TEXT"),
    ("element_vectors", "mean_json", "TEXT"),
)


@contextmanager
def connect(database_path: str) -> Iterator[sqlite3.Connection]:
    """Соединение на одно действие: транзакция фиксируется при выходе без ошибки и откатывается при
    ошибке, а само соединение закрывается. Прежде with sqlite3.connect() только фиксировал
    транзакцию, а соединение оставалось открытым до сборки мусора и держало файл."""
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def ensure_schema(database_path: str) -> None:
    """Создает таблицы, включает WAL и доводит схему до текущей версии."""
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        # Режим журнала меняется только вне транзакции; база в памяти отвечает «memory», и это не ошибка
        connection.execute("PRAGMA journal_mode=WAL")
        if connection.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return
        connection.execute("BEGIN")
        try:
            for statement in BASE_SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            for table, column, definition in _COLUMNS:
                # Колонка добавляется, только если ее нет: база могла дойти до середины миграции
                present = connection.execute(
                    "SELECT COUNT(*) FROM pragma_table_info(?) WHERE name = ?", (table, column)).fetchone()[0]
                if not present:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            connection.execute("CREATE INDEX IF NOT EXISTS ix_rounds_feedback ON rounds (feedback_type, id)")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
