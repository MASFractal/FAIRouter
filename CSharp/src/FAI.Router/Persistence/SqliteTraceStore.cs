using System.Text.Json;
using FAI.Router.Enums;
using AI.DataStructs.Algebraic;
using FAI.Router.JudgeLogic;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using Microsoft.Data.Sqlite;

namespace FAI.Router.Persistence;

/// <summary>
/// Ход, накопленный для обучения: трассировка, отзыв и пара «заказ / факт»
/// </summary>
/// <param name="Id">Идентификатор хода в базе</param>
/// <param name="Trace">Трассировка: победитель, соперники, признаки задачи</param>
/// <param name="Feedback">Отзыв на ответ победителя</param>
/// <param name="Requested">Заказанная спецификация; null, если ход записан без неё</param>
/// <param name="Actual">Фактическая спецификация ответа; null, если ответ не замерялся</param>
public record TrainingRound(
    long Id,
    Tracert Trace,
    Feedback Feedback,
    Specifications? Requested,
    Specifications? Actual);

/// <summary>
/// Накопитель ходов и отзывов в той же базе, что и веса. Тренеры делают шаг по одному примеру,
/// а закономерность видна только на выборке — накопитель и есть то, из чего она берётся.
/// Отзыв приходит позже самого хода, поэтому пишется отдельным действием.
/// </summary>
public class SqliteTraceStore
{
    private readonly string _databasePath;

    /// <summary>
    /// Накопитель ходов и отзывов
    /// </summary>
    /// <param name="databasePath">Путь к файлу базы, создаётся при отсутствии</param>
    public SqliteTraceStore(string databasePath)
    {
        _databasePath = databasePath;
        EnsureSchema();
    }

    /// <summary>
    /// Записывает ход. Возвращает идентификатор, под которым позже проставляется отзыв.
    /// </summary>
    /// <param name="trace">Трассировка хода</param>
    /// <param name="requested">Заказанная спецификация</param>
    /// <param name="actual">Фактическая спецификация ответа</param>
    /// <param name="prompt">Текст запроса, для разбора глазами</param>
    public long Append(Tracert trace, Specifications? requested = null, Specifications? actual = null, string? prompt = null)
    {
        if (string.IsNullOrWhiteSpace(trace.Winner.Name))
            throw new InvalidOperationException("Победитель без имени: по такому ходу кандидата потом не опознать.");

        string[] rivals = [.. trace.TopKElements
            .Where(element => element != trace.Winner)
            .Select(element => element.Name ?? "")];

        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            INSERT INTO rounds (created_at, prompt, winner, rivals_json, features_json, score, requested_json, actual_json)
            VALUES ($createdAt, $prompt, $winner, $rivals, $features, $score, $requested, $actual);
            SELECT last_insert_rowid();
            """;
        command.Parameters.AddWithValue("$createdAt", DateTime.UtcNow.ToString("O"));
        command.Parameters.AddWithValue("$prompt", (object?)prompt ?? DBNull.Value);
        command.Parameters.AddWithValue("$winner", trace.Winner.Name);
        command.Parameters.AddWithValue("$rivals", JsonSerializer.Serialize(rivals));
        command.Parameters.AddWithValue("$features", SerializeVector(trace.InputFeatureVector));
        command.Parameters.AddWithValue("$score", trace.Score);
        command.Parameters.AddWithValue("$requested", (object?)Serialize(requested) ?? DBNull.Value);
        command.Parameters.AddWithValue("$actual", (object?)Serialize(actual) ?? DBNull.Value);

        return (long)command.ExecuteScalar()!;
    }

    /// <summary>
    /// Проставляет отзыв к записанному ходу
    /// </summary>
    /// <param name="roundId">Идентификатор хода</param>
    /// <param name="feedback">Отзыв</param>
    public void SetFeedback(long roundId, Feedback feedback)
    {
        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "UPDATE rounds SET feedback_type = $type, feedback_score = $score WHERE id = $id";
        command.Parameters.AddWithValue("$type", (int)feedback.FType);
        command.Parameters.AddWithValue("$score", feedback.FeadbackScore);
        command.Parameters.AddWithValue("$id", roundId);

        if (command.ExecuteNonQuery() == 0)
            throw new InvalidOperationException($"Хода {roundId} в базе нет: отзыв не к чему привязать.");
    }

    /// <summary>
    /// Обучающая выборка: ходы, у которых есть отзыв, свежие первыми.
    /// Ход пропускается, если кто-то из его участников больше не значится в каталоге —
    /// снятую модель незачем ни поощрять, ни наказывать.
    /// </summary>
    /// <param name="catalog">Действующие кандидаты, по именам которых восстанавливается трассировка</param>
    /// <param name="limit">Сколько ходов взять</param>
    public IReadOnlyList<TrainingRound> ReadRated(IEnumerable<BaseRoutedElement> catalog, int limit = 1000)
    {
        Dictionary<string, BaseRoutedElement> byName = catalog
            .Where(element => !string.IsNullOrWhiteSpace(element.Name))
            .ToDictionary(element => element.Name!);

        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT id, winner, rivals_json, features_json, score, requested_json, actual_json, feedback_type, feedback_score
            FROM rounds
            WHERE feedback_score IS NOT NULL
            ORDER BY id DESC
            LIMIT $limit
            """;
        command.Parameters.AddWithValue("$limit", limit);

        List<TrainingRound> rounds = [];
        using SqliteDataReader reader = command.ExecuteReader();

        while (reader.Read())
        {
            if (!byName.TryGetValue(reader.GetString(1), out BaseRoutedElement? winner))
                continue;

            string[] rivalNames = JsonSerializer.Deserialize<string[]>(reader.GetString(2)) ?? [];

            if (!rivalNames.All(byName.ContainsKey))
                continue;

            Tracert trace = new()
            {
                Winner = winner,
                TopKElements = [winner, .. rivalNames.Select(name => byName[name])],
                InputFeatureVector = new Vector(JsonSerializer.Deserialize<double[]>(reader.GetString(3)) ?? []),
                Score = reader.GetDouble(4)
            };

            Feedback feedback = new()
            {
                FType = (FeedbackType)reader.GetInt32(7),
                FeadbackScore = reader.GetDouble(8)
            };

            rounds.Add(new TrainingRound(
                reader.GetInt64(0),
                trace,
                feedback,
                Deserialize(reader, 5),
                Deserialize(reader, 6)));
        }

        return rounds;
    }

    /// <summary>
    /// Сколько ходов записано и сколько из них уже оценено
    /// </summary>
    public (int Total, int Rated) Count()
    {
        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT COUNT(*), COUNT(feedback_score) FROM rounds";

        using SqliteDataReader reader = command.ExecuteReader();
        reader.Read();

        return (reader.GetInt32(0), reader.GetInt32(1));
    }

    private static string? Serialize(Specifications? specifications) =>
        specifications is null ? null : JsonSerializer.Serialize(specifications);

    private static Specifications? Deserialize(SqliteDataReader reader, int column) =>
        reader.IsDBNull(column) ? null : JsonSerializer.Deserialize<Specifications>(reader.GetString(column));

    private static string SerializeVector(Vector vector)
    {
        double[] values = new double[vector.Count];

        for (int i = 0; i < vector.Count; i++)
            values[i] = vector[i];

        return JsonSerializer.Serialize(values);
    }

    private void EnsureSchema()
    {
        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            CREATE TABLE IF NOT EXISTS rounds (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     TEXT    NOT NULL,
                prompt         TEXT,
                winner         TEXT    NOT NULL,
                rivals_json    TEXT    NOT NULL,
                features_json  TEXT    NOT NULL,
                score          REAL    NOT NULL DEFAULT 0,
                requested_json TEXT,
                actual_json    TEXT,
                feedback_type  INTEGER,
                feedback_score REAL
            );
            """;
        command.ExecuteNonQuery();
    }
}
