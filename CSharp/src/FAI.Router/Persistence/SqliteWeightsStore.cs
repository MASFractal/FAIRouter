using System.Text.Json;
using Microsoft.Data.Sqlite;
using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;
using FAI.Router.RoutedElements;


namespace FAI.Router.Persistence;

/// <summary>
/// Хранилище обученных весов в SQLite: векторы соответствия кандидатов и матрица судьи.
/// Кандидат опознаётся по имени — переименование равносильно потере обучения.
/// </summary>
public class SqliteWeightsStore
{
    private readonly string _databasePath;

    /// <summary>
    /// Хранилище обученных весов в SQLite
    /// </summary>
    /// <param name="databasePath">Путь к файлу базы, создаётся при отсутствии</param>
    public SqliteWeightsStore(string databasePath)
    {
        _databasePath = databasePath;
        EnsureSchema();
    }

    /// <summary>
    /// Сохраняет векторы соответствия кандидатов
    /// </summary>
    /// <param name="elements">Кандидаты</param>
    public void Save(IEnumerable<BaseRoutedElement> elements)
    {
        using SqliteConnection connection = Open();
        using SqliteTransaction transaction = connection.BeginTransaction();

        foreach (BaseRoutedElement element in elements)
        {
            if (string.IsNullOrWhiteSpace(element.Name))
                throw new InvalidOperationException("Кандидат без имени: вес не под чем сохранять.");

            using SqliteCommand command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText =
                """
                INSERT INTO element_vectors (name, dimension, values_json) VALUES ($name, $dimension, $values)
                ON CONFLICT(name) DO UPDATE SET dimension = excluded.dimension, values_json = excluded.values_json
                """;
            command.Parameters.AddWithValue("$name", element.Name);
            command.Parameters.AddWithValue("$dimension", element.IdealMatchVector.Count);
            command.Parameters.AddWithValue("$values", Serialize(element.IdealMatchVector));
            command.ExecuteNonQuery();
        }

        transaction.Commit();
    }

    /// <summary>
    /// Восстанавливает векторы соответствия. Возвращает число узнанных кандидатов;
    /// незнакомые остаются с начальными весами.
    /// </summary>
    /// <param name="elements">Кандидаты</param>
    public int Load(IEnumerable<BaseRoutedElement> elements)
    {
        using SqliteConnection connection = Open();
        int restored = 0;

        foreach (BaseRoutedElement element in elements)
        {
            using SqliteCommand command = connection.CreateCommand();
            command.CommandText = "SELECT values_json FROM element_vectors WHERE name = $name";
            command.Parameters.AddWithValue("$name", element.Name ?? "");

            if (command.ExecuteScalar() is not string json)
                continue;

            double[] values = Deserialize(json);
            Expect(values.Length, element.IdealMatchVector.Count, $"вектор кандидата «{element.Name}»");

            element.IdealMatchVector = new Vector(values);
            restored++;
        }

        return restored;
    }

    /// <summary>
    /// Сохраняет матрицу трансформации судьи
    /// </summary>
    /// <param name="judge">Судья</param>
    public void Save(Judge judge)
    {
        Matrix matrix = judge.TransformerW;
        double[] values = new double[matrix.Height * matrix.Width];

        for (int row = 0; row < matrix.Height; row++)
            for (int column = 0; column < matrix.Width; column++)
                values[row * matrix.Width + column] = matrix[row, column];

        using SqliteConnection connection = Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            INSERT INTO judge_matrix (id, rows_count, columns_count, values_json) VALUES (1, $rows, $columns, $values)
            ON CONFLICT(id) DO UPDATE SET rows_count = excluded.rows_count,
                                          columns_count = excluded.columns_count,
                                          values_json = excluded.values_json
            """;
        command.Parameters.AddWithValue("$rows", matrix.Height);
        command.Parameters.AddWithValue("$columns", matrix.Width);
        command.Parameters.AddWithValue("$values", JsonSerializer.Serialize(values));
        command.ExecuteNonQuery();

        connection.Close();
    }

    /// <summary>
    /// Восстанавливает матрицу судьи. Ложь — обученной матрицы в базе ещё нет.
    /// </summary>
    /// <param name="judge">Судья</param>
    public bool Load(Judge judge)
    {
        using SqliteConnection connection = Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT rows_count, columns_count, values_json FROM judge_matrix WHERE id = 1";

        using SqliteDataReader reader = command.ExecuteReader();

        if (!reader.Read())
            return false;

        int rows = reader.GetInt32(0);
        int columns = reader.GetInt32(1);
        double[] values = Deserialize(reader.GetString(2));

        Expect(rows, judge.TransformerW.Height, "число строк матрицы судьи");
        Expect(columns, judge.TransformerW.Width, "число столбцов матрицы судьи");

        Matrix matrix = new(rows, columns);

        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                matrix[row, column] = values[row * columns + column];

        judge.TransformerW = matrix;

        return true;
    }

    // Размерность признаков меняется при добавлении стиля или метрики: молча взятые
    // веса прежней длины означали бы обучение поверх мусора
    private static void Expect(int stored, int expected, string what)
    {
        if (stored != expected)
            throw new InvalidOperationException(
                $"В базе {what} размерности {stored}, а сейчас ожидается {expected}. " +
                "Признаки изменились — сохранённые веса больше не применимы.");
    }

    private static string Serialize(Vector vector)
    {
        double[] values = new double[vector.Count];

        for (int i = 0; i < vector.Count; i++)
            values[i] = vector[i];

        return JsonSerializer.Serialize(values);
    }

    private static double[] Deserialize(string json) =>
        JsonSerializer.Deserialize<double[]>(json) ?? [];

    private SqliteConnection Open() => SqliteDb.Open(_databasePath);

    private void EnsureSchema()
    {
        using SqliteConnection connection = Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            CREATE TABLE IF NOT EXISTS element_vectors (
                name        TEXT    PRIMARY KEY,
                dimension   INTEGER NOT NULL,
                values_json TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS judge_matrix (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                rows_count    INTEGER NOT NULL,
                columns_count INTEGER NOT NULL,
                values_json   TEXT    NOT NULL
            );
            """;
        command.ExecuteNonQuery();
    }
}
