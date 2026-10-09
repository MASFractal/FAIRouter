using System.Text.Json;
using Microsoft.Data.Sqlite;
using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;
using FAI.Router.RoutedElements;

namespace FAI.Router.Persistence;

/// <summary>
/// Хранилище обученных весов в SQLite: векторы соответствия кандидатов вместе с их средним задач и
/// матрица судьи. Кандидат опознается по имени, поэтому переименование равносильно потере обучения.
/// </summary>
/// <remarks>
/// Сохранять стоит только векторы, которых касалось обучение: вектор, записанный необученным,
/// при загрузке затирал бы свежий начальный вектор по новому снимку рейтингов. Это решает
/// вызывающий, хранилище пишет то, что дали.
/// </remarks>
public class SqliteWeightsStore
{
    private readonly string _databasePath;

    /// <summary>
    /// Хранилище обученных весов в SQLite
    /// </summary>
    /// <param name="databasePath">Путь к файлу базы, создается при отсутствии</param>
    public SqliteWeightsStore(string databasePath)
    {
        _databasePath = databasePath;
        SqliteDb.EnsureSchema(databasePath);
    }

    /// <summary>
    /// Сохраняет векторы соответствия кандидатов одной транзакцией
    /// </summary>
    /// <param name="elements">Кандидаты</param>
    public void Save(IEnumerable<BaseRoutedElement> elements) => Save(elements, null);

    /// <summary>
    /// Сохраняет векторы кандидатов и матрицу судьи одной транзакцией: база не остается с новыми
    /// векторами и старой матрицей, если запись прервалась посередине
    /// </summary>
    /// <param name="elements">Кандидаты</param>
    /// <param name="judge">Судья; пусто, значит только векторы</param>
    public void Save(IEnumerable<BaseRoutedElement> elements, Judge? judge)
    {
        using SqliteConnection connection = Open();
        using SqliteTransaction transaction = connection.BeginTransaction();

        foreach (BaseRoutedElement element in elements)
            SaveVector(connection, transaction, element);

        if (judge is not null)
            SaveJudge(connection, transaction, judge);

        transaction.Commit();
    }

    /// <summary>
    /// Восстанавливает векторы соответствия. Возвращает число узнанных кандидатов;
    /// незнакомые остаются с начальными весами.
    /// </summary>
    /// <param name="elements">Кандидаты</param>
    public int Load(IEnumerable<BaseRoutedElement> elements) => LoadVectors(elements).Count;

    /// <summary>
    /// Восстанавливает векторы соответствия и их среднее задач. Возвращает имена узнанных кандидатов.
    /// </summary>
    /// <remarks>
    /// Вектор другой размерности пропускается с предупреждением, а не роняет загрузку всех: признаки
    /// сменили длину, и обучение этого кандидата к ним не относится, а остальные от этого не портятся.
    /// Вектор, сохраненный до того, как среднее стало храниться при нем, получает общее среднее из
    /// таблицы task_mean: именно в нем его тогда учили.
    /// </remarks>
    /// <param name="elements">Кандидаты</param>
    public IReadOnlyList<string> LoadVectors(IEnumerable<BaseRoutedElement> elements)
    {
        using SqliteConnection connection = Open();
        Vector? legacyMean = ReadTaskMean(connection);
        List<string> restored = [];

        foreach (BaseRoutedElement element in elements)
        {
            using SqliteCommand command = connection.CreateCommand();
            command.CommandText = "SELECT values_json, mean_json FROM element_vectors WHERE name = $name";
            command.Parameters.AddWithValue("$name", element.Name ?? "");

            using SqliteDataReader reader = command.ExecuteReader();

            if (!reader.Read())
                continue;

            double[] values = Deserialize(reader.GetString(0));
            Vector? mean = reader.IsDBNull(1) ? legacyMean : new Vector(Deserialize(reader.GetString(1)));

            if (values.Length != element.IdealMatchVector.Count || !values.All(double.IsFinite)
                || (mean is not null && mean.Count != values.Length))
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"FAIRouter: вектор кандидата «{element.Name}» в базе размерности {values.Length}, а ожидается " +
                    $"{element.IdealMatchVector.Count}, либо он поврежден; кандидат остается с начальными весами.");
                continue;
            }

            element.IdealMatchVector = new Vector(values);
            element.TaskMean = mean;
            restored.Add(element.Name!);
        }

        return restored;
    }

    /// <summary>
    /// Сохраняет матрицу трансформации судьи
    /// </summary>
    /// <param name="judge">Судья</param>
    public void Save(Judge judge)
    {
        using SqliteConnection connection = Open();
        SaveJudge(connection, null, judge);
    }

    /// <summary>
    /// Восстанавливает матрицу судьи. Ложь означает, что обученной матрицы в базе нет или она
    /// другой размерности (тогда с предупреждением остается нынешняя).
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

        if (rows != judge.TransformerW.Height || columns != judge.TransformerW.Width || values.Length != rows * columns)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"FAIRouter: матрица судьи в базе {rows}x{columns}, а ожидается {judge.TransformerW.Height}x{judge.TransformerW.Width}; " +
                "признаки изменились, судья остается с нынешней матрицей.");
            return false;
        }

        Matrix matrix = new(rows, columns);

        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                matrix[row, column] = values[row * columns + column];

        judge.TransformerW = matrix;

        return true;
    }

    /// <summary>
    /// Сохраняет общее среднее по векторам задач. Оставлено ради совместимости: среднее теперь
    /// хранится при каждом векторе.
    /// </summary>
    /// <param name="taskMean">Среднее</param>
    public void SaveTaskMean(Vector taskMean)
    {
        using SqliteConnection connection = Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            INSERT INTO task_mean (id, values_json) VALUES (1, $values)
            ON CONFLICT(id) DO UPDATE SET values_json = excluded.values_json
            """;
        command.Parameters.AddWithValue("$values", Serialize(taskMean));
        command.ExecuteNonQuery();
    }

    /// <summary>
    /// Восстанавливает общее среднее по векторам задач. Пусто, если его не сохраняли или оно другой
    /// размерности, чем нынешние признаки.
    /// </summary>
    public Vector? LoadTaskMean()
    {
        using SqliteConnection connection = Open();

        return ReadTaskMean(connection);
    }

    private static Vector? ReadTaskMean(SqliteConnection connection)
    {
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT values_json FROM task_mean WHERE id = 1";

        if (command.ExecuteScalar() is not string json)
            return null;

        double[] values = Deserialize(json);

        if (values.Length == Settings.VectorDim)
            return new Vector(values);

        System.Diagnostics.Trace.TraceWarning(
            $"FAIRouter: среднее задач в базе размерности {values.Length}, а ожидается {Settings.VectorDim}; оно не применяется.");
        return null;
    }

    private static void SaveVector(SqliteConnection connection, SqliteTransaction transaction, BaseRoutedElement element)
    {
        if (string.IsNullOrWhiteSpace(element.Name))
            throw new InvalidOperationException("Кандидат без имени: вес не под чем сохранять.");

        using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText =
            """
            INSERT INTO element_vectors (name, dimension, values_json, mean_json) VALUES ($name, $dimension, $values, $mean)
            ON CONFLICT(name) DO UPDATE SET dimension = excluded.dimension, values_json = excluded.values_json, mean_json = excluded.mean_json
            """;
        command.Parameters.AddWithValue("$name", element.Name);
        command.Parameters.AddWithValue("$dimension", element.IdealMatchVector.Count);
        command.Parameters.AddWithValue("$values", Serialize(element.IdealMatchVector));
        command.Parameters.AddWithValue("$mean", (element.TaskMean ?? Settings.TaskMean) is { } mean ? Serialize(mean) : DBNull.Value);
        command.ExecuteNonQuery();
    }

    private static void SaveJudge(SqliteConnection connection, SqliteTransaction? transaction, Judge judge)
    {
        Matrix matrix = judge.TransformerW;
        double[] values = new double[matrix.Height * matrix.Width];

        for (int row = 0; row < matrix.Height; row++)
            for (int column = 0; column < matrix.Width; column++)
                values[row * matrix.Width + column] = matrix[row, column];

        using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
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
    }

    private static string Serialize(Vector vector) => JsonSerializer.Serialize(vector.ToArray());

    private static double[] Deserialize(string json) =>
        JsonSerializer.Deserialize<double[]>(json) ?? [];

    private SqliteConnection Open() => SqliteDb.Open(_databasePath);
}
