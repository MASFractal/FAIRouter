using Microsoft.Data.Sqlite;

namespace FAI.Router.Persistence;

/// <summary>
/// Открытие базы и ее схема: веса и накопленный опыт лежат в одном файле, поэтому и путь к нему
/// превращается в соединение одинаково, и схема у обоих хранилищ одна
/// </summary>
/// <remarks>
/// Версия схемы хранится в PRAGMA user_version. База прежней версии доводится до текущей
/// добавлением колонок, а не пересозданием: журнал и веса в ней остаются. Журнал пишется в режиме
/// WAL, чтобы запись хода не ждала чтения обучения и наоборот.
/// </remarks>
internal static class SqliteDb
{
    /// <summary>
    /// Версия схемы. 1: таблицы журнала, весов, среднего и судьи. 2: прогноз в момент выбора, отметка
    /// обученного хода и отказавшие кандидаты в журнале, среднее задач у каждого вектора.
    /// </summary>
    private const int SchemaVersion = 2;

    private const string BaseSchema =
        """
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
        """;

    /// <summary>
    /// Открытое соединение с базой; файл создается при отсутствии
    /// </summary>
    /// <param name="databasePath">Путь к файлу базы</param>
    public static SqliteConnection Open(string databasePath)
    {
        SqliteConnection connection = new(new SqliteConnectionStringBuilder { DataSource = databasePath }.ToString());
        connection.Open();

        return connection;
    }

    /// <summary>
    /// Создает таблицы, включает WAL и доводит схему до текущей версии
    /// </summary>
    /// <param name="databasePath">Путь к файлу базы</param>
    public static void EnsureSchema(string databasePath)
    {
        using SqliteConnection connection = Open(databasePath);

        // Режим журнала меняется только вне транзакции; база в памяти отвечает «memory», и это не ошибка
        Execute(connection, null, "PRAGMA journal_mode=WAL;");

        if (Version(connection) >= SchemaVersion)
            return;

        using SqliteTransaction transaction = connection.BeginTransaction();

        Execute(connection, transaction, BaseSchema);
        AddColumn(connection, transaction, "rounds", "predicted_quality", "REAL");
        AddColumn(connection, transaction, "rounds", "trained", "INTEGER NOT NULL DEFAULT 0");
        AddColumn(connection, transaction, "rounds", "failed_json", "TEXT");
        AddColumn(connection, transaction, "element_vectors", "mean_json", "TEXT");
        Execute(connection, transaction, "CREATE INDEX IF NOT EXISTS ix_rounds_feedback ON rounds (feedback_type, id);");
        Execute(connection, transaction, $"PRAGMA user_version = {SchemaVersion};");

        transaction.Commit();
    }

    /// <summary>Команда без результата</summary>
    public static void Execute(SqliteConnection connection, SqliteTransaction? transaction, string sql)
    {
        using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = sql;
        command.ExecuteNonQuery();
    }

    private static int Version(SqliteConnection connection)
    {
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "PRAGMA user_version;";

        return Convert.ToInt32(command.ExecuteScalar());
    }

    // Колонка добавляется, только если ее нет: база могла дойти до середины миграции
    private static void AddColumn(SqliteConnection connection, SqliteTransaction transaction, string table, string column, string definition)
    {
        using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = $"SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = $column;";
        command.Parameters.AddWithValue("$column", column);

        if (Convert.ToInt32(command.ExecuteScalar()) == 0)
            Execute(connection, transaction, $"ALTER TABLE {table} ADD COLUMN {column} {definition};");
    }
}
