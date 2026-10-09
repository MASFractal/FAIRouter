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
/// <param name="Requested">Заказанная спецификация; null, если ход записан без нее</param>
/// <param name="Actual">Фактическая спецификация ответа; null, если ответ не замерялся</param>
public record TrainingRound(
    long Id,
    Tracert Trace,
    Feedback Feedback,
    Specifications? Requested,
    Specifications? Actual);

/// <summary>
/// Накопитель ходов и отзывов в той же базе, что и веса. Тренеры делают шаг по одному примеру,
/// а закономерность видна только на выборке, а накопитель и есть то, из чего она берется.
/// Отзыв приходит позже самого хода, поэтому пишется отдельным действием.
/// </summary>
/// <remarks>
/// Журнал хранит тексты запросов. Срок их хранения задает владелец базы: <see cref="Purge"/>
/// удаляет старые ходы целиком либо только их тексты.
/// </remarks>
public class SqliteTraceStore
{
    private const string RoundColumns =
        "id, winner, rivals_json, features_json, score, requested_json, actual_json, feedback_type, feedback_score, is_exploration, predicted_quality";

    private readonly string _databasePath;

    /// <summary>
    /// Накопитель ходов и отзывов
    /// </summary>
    /// <param name="databasePath">Путь к файлу базы, создается при отсутствии</param>
    public SqliteTraceStore(string databasePath)
    {
        _databasePath = databasePath;
        SqliteDb.EnsureSchema(databasePath);
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
            INSERT INTO rounds (created_at, prompt, winner, rivals_json, features_json, score, is_exploration, requested_json, actual_json, predicted_quality, failed_json)
            VALUES ($createdAt, $prompt, $winner, $rivals, $features, $score, $exploration, $requested, $actual, $forecast, $failed);
            SELECT last_insert_rowid();
            """;
        command.Parameters.AddWithValue("$createdAt", DateTime.UtcNow.ToString("O"));
        command.Parameters.AddWithValue("$prompt", (object?)prompt ?? DBNull.Value);
        command.Parameters.AddWithValue("$winner", trace.Winner.Name);
        command.Parameters.AddWithValue("$rivals", JsonSerializer.Serialize(rivals));
        command.Parameters.AddWithValue("$features", SerializeVector(trace.InputFeatureVector));
        command.Parameters.AddWithValue("$score", trace.Score);
        command.Parameters.AddWithValue("$exploration", trace.IsExploration ? 1 : 0);
        command.Parameters.AddWithValue("$requested", (object?)Serialize(requested) ?? DBNull.Value);
        command.Parameters.AddWithValue("$actual", (object?)Serialize(actual) ?? DBNull.Value);
        command.Parameters.AddWithValue("$forecast", trace.Forecast is { } forecast && double.IsFinite(forecast) ? forecast : DBNull.Value);
        command.Parameters.AddWithValue("$failed", trace.Failed.Count > 0 ? JsonSerializer.Serialize(trace.Failed) : DBNull.Value);

        return (long)command.ExecuteScalar()!;
    }

    /// <summary>
    /// Проставляет отзыв к записанному ходу. Ход снова попадает в очередь обучения: новый отзыв
    /// (человек поверх автоотзыва) должен переучить его своим знаком.
    /// </summary>
    /// <param name="roundId">Идентификатор хода</param>
    /// <param name="feedback">Отзыв</param>
    public void SetFeedback(long roundId, Feedback feedback)
    {
        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "UPDATE rounds SET feedback_type = $type, feedback_score = $score, trained = 0 WHERE id = $id";
        command.Parameters.AddWithValue("$type", (int)feedback.FType);
        command.Parameters.AddWithValue("$score", feedback.FeadbackScore);
        command.Parameters.AddWithValue("$id", roundId);

        if (command.ExecuteNonQuery() == 0)
            throw new InvalidOperationException($"Хода {roundId} в базе нет: отзыв не к чему привязать.");
    }

    /// <summary>
    /// Обучающая выборка: последние limit ходов с отзывом, по порядку записи (старые первыми), чтобы
    /// последнее слово в обучении было за свежими. Победитель, снятый с каталога, выбрасывает ход:
    /// снятую модель незачем ни поощрять, ни наказывать. Снятый соперник выпадает только из пары.
    /// </summary>
    /// <param name="catalog">Действующие кандидаты, по именам которых восстанавливается трассировка</param>
    /// <param name="limit">Сколько ходов взять</param>
    /// <param name="untrainedOnly">Только ходы, которые еще не учили (<see cref="MarkTrained"/>)</param>
    public IReadOnlyList<TrainingRound> ReadRated(IEnumerable<BaseRoutedElement> catalog, int limit = 1000, bool untrainedOnly = false)
    {
        Dictionary<string, BaseRoutedElement> byName = catalog.ByName();

        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            $"""
            SELECT {RoundColumns} FROM (
                SELECT {RoundColumns}
                FROM rounds
                WHERE feedback_score IS NOT NULL {(untrainedOnly ? "AND trained = 0" : "")}
                ORDER BY id DESC
                LIMIT $limit)
            ORDER BY id
            """;
        command.Parameters.AddWithValue("$limit", limit);

        List<TrainingRound> rounds = [];
        using SqliteDataReader reader = command.ExecuteReader();

        while (reader.Read())
            if (Round(reader, byName) is { } round)
                rounds.Add(round);

        return rounds;
    }

    /// <summary>
    /// Пары для калибровки планки: прогноз качества победителя В МОМЕНТ ВЫБОРА и оценка человека,
    /// последние limit человеческих отзывов. Автоотзывы не берутся: планка обещает вероятность лайка
    /// человека, а не согласие судьи с самим собой.
    /// </summary>
    /// <remarks>
    /// Прогноз при нынешних весах уже видел этот отзыв в обучении и обещал бы больше, чем знает. Ходы,
    /// записанные до появления колонки прогноза, берут прогноз нынешнего вектора победителя.
    /// </remarks>
    /// <param name="catalog">Действующие кандидаты, для ходов без записанного прогноза</param>
    /// <param name="limit">Сколько последних человеческих отзывов взять</param>
    public IReadOnlyList<(double Quality, double Score)> ReadCalibration(IEnumerable<BaseRoutedElement> catalog, int limit = 1000)
    {
        Dictionary<string, BaseRoutedElement> byName = catalog.ByName();

        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT predicted_quality, winner, features_json, feedback_score
            FROM rounds
            WHERE feedback_type = $human AND feedback_score IS NOT NULL
            ORDER BY id DESC
            LIMIT $limit
            """;
        command.Parameters.AddWithValue("$human", (int)FeedbackType.Human);
        command.Parameters.AddWithValue("$limit", limit);

        List<(double, double)> pairs = [];
        using SqliteDataReader reader = command.ExecuteReader();

        while (reader.Read())
        {
            double? quality = !reader.IsDBNull(0)
                ? reader.GetDouble(0)
                : byName.TryGetValue(reader.GetString(1), out BaseRoutedElement? winner)
                    ? winner.GetQualityScore(DeserializeVector(reader.GetString(2)))
                    : null;

            if (quality is { } known && double.IsFinite(known))
                pairs.Add((known, reader.GetDouble(3)));
        }

        return pairs;
    }

    /// <summary>
    /// Отмечает ходы обученными: следующее обучение их не повторит, пока к ним не придет новый отзыв
    /// </summary>
    /// <param name="roundIds">Идентификаторы ходов</param>
    public void MarkTrained(IEnumerable<long> roundIds)
    {
        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteTransaction transaction = connection.BeginTransaction();
        using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = "UPDATE rounds SET trained = 1 WHERE id = $id";
        SqliteParameter id = command.Parameters.Add("$id", SqliteType.Integer);

        foreach (long roundId in roundIds)
        {
            id.Value = roundId;
            command.ExecuteNonQuery();
        }

        transaction.Commit();
    }

    /// <summary>
    /// Заполняет опыт кандидатов из журнала: условное число оцененных ходов и оценку дисперсии
    /// отзывов. Возвращает число узнанных кандидатов.
    /// </summary>
    /// <remarks>
    /// Эти две величины нужны формуле температуры выбора, и копить их отдельно не требуется:
    /// в журнале уже лежит каждый ход с победителем и оценкой отзыва.
    /// <para>
    /// Человеческий отзыв идет в опыт целиком, автоотзыв с весом <see cref="Settings.AutoFeedbackWeight"/>.
    /// Прежде автоотзывы не считались вовсе, и без человеческих отзывов разведка не остывала никогда;
    /// еще раньше они считались наравне, и разведка гасла по мнению собственного судьи.
    /// </para>
    /// <para>
    /// Дисперсия стягивается к неизвестной (<see cref="Settings.UnknownVariance"/>) с силой
    /// <see cref="Settings.VariancePriorStrength"/>: два одинаковых отзыва давали нулевую дисперсию и
    /// уверенность на пустом месте.
    /// </para>
    /// </remarks>
    /// <param name="elements">Кандидаты</param>
    public int LoadStatistics(IEnumerable<BaseRoutedElement> elements)
    {
        Dictionary<string, BaseRoutedElement> byName = elements.ByName();

        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT winner, SUM(w), SUM(w * feedback_score), SUM(w * feedback_score * feedback_score)
            FROM (SELECT winner, feedback_score, CASE WHEN feedback_type = $human THEN 1.0 ELSE $auto END AS w
                  FROM rounds WHERE feedback_score IS NOT NULL)
            GROUP BY winner
            """;
        command.Parameters.AddWithValue("$human", (int)FeedbackType.Human);
        command.Parameters.AddWithValue("$auto", Settings.AutoFeedbackWeight);

        using SqliteDataReader reader = command.ExecuteReader();
        int restored = 0;

        while (reader.Read())
        {
            if (!byName.TryGetValue(reader.GetString(0), out BaseRoutedElement? element))
                continue;

            double weight = reader.GetDouble(1);
            double mean = weight > 0 ? reader.GetDouble(2) / weight : 0;
            double spread = weight > 0 ? Math.Max(0, reader.GetDouble(3) / weight - mean * mean) : 0;
            double strength = Settings.VariancePriorStrength;

            element.Experience = weight;
            element.ScoreVariance = (strength * Settings.UnknownVariance + weight * spread) / (strength + Math.Max(weight - 1, 0));
            restored++;
        }

        return restored;
    }

    /// <summary>
    /// Среднее по векторам задач последних limit ходов. Пусто, если ходов еще нет.
    /// </summary>
    /// <remarks>
    /// Ходы хранятся несмещенными именно ради этого расчета: если бы в журнал попадали уже
    /// смещенные векторы, среднее считалось бы само из себя и с каждым пересчетом уползало.
    /// Векторы другой длины (журнал прежней размерности признаков) пропускаются.
    /// </remarks>
    /// <param name="limit">Сколько последних ходов взять</param>
    public Vector? GetFeatureMean(int limit = 1000)
    {
        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "SELECT features_json FROM rounds ORDER BY id DESC LIMIT $limit";
        command.Parameters.AddWithValue("$limit", limit);

        using SqliteDataReader reader = command.ExecuteReader();

        Vector? sum = null;
        int count = 0;

        while (reader.Read())
        {
            Vector features = DeserializeVector(reader.GetString(0));

            if (sum is not null && features.Count != sum.Count)
                continue;

            sum = sum is null ? features : sum + features;
            count++;
        }

        return count == 0 ? null : sum! / count;
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

    /// <summary>
    /// Срок хранения журнала: удаляет ходы, записанные раньше <paramref name="before"/>, либо только
    /// стирает их тексты запросов, оставляя признаки и отзывы для обучения. Возвращает число ходов.
    /// </summary>
    /// <param name="before">Граница: старше нее ходы чистятся</param>
    /// <param name="promptsOnly">Стереть только тексты запросов, а сами ходы оставить</param>
    public int Purge(DateTimeOffset before, bool promptsOnly = false)
    {
        using SqliteConnection connection = SqliteDb.Open(_databasePath);
        using SqliteCommand command = connection.CreateCommand();

        // Метка записи в формате «O» по UTC, поэтому строки сравниваются в порядке времени
        command.CommandText = promptsOnly
            ? "UPDATE rounds SET prompt = NULL WHERE created_at < $before AND prompt IS NOT NULL"
            : "DELETE FROM rounds WHERE created_at < $before";
        command.Parameters.AddWithValue("$before", before.UtcDateTime.ToString("O"));

        return command.ExecuteNonQuery();
    }

    private static TrainingRound? Round(SqliteDataReader reader, Dictionary<string, BaseRoutedElement> byName)
    {
        if (!byName.TryGetValue(reader.GetString(1), out BaseRoutedElement? winner))
            return null;

        string[] rivalNames = JsonSerializer.Deserialize<string[]>(reader.GetString(2)) ?? [];

        Tracert trace = new()
        {
            Winner = winner,
            TopKElements = [winner, .. rivalNames.Where(byName.ContainsKey).Select(name => byName[name]).Where(rival => rival != winner)],
            InputFeatureVector = DeserializeVector(reader.GetString(3)),
            Score = reader.GetDouble(4),
            IsExploration = reader.GetInt32(9) != 0,
            Forecast = reader.IsDBNull(10) ? null : reader.GetDouble(10)
        };

        Feedback feedback = new()
        {
            FType = (FeedbackType)reader.GetInt32(7),
            FeadbackScore = reader.GetDouble(8)
        };

        return new TrainingRound(reader.GetInt64(0), trace, feedback, Deserialize(reader, 5), Deserialize(reader, 6));
    }

    private static string? Serialize(Specifications? specifications) =>
        specifications is null ? null : JsonSerializer.Serialize(specifications);

    private static Specifications? Deserialize(SqliteDataReader reader, int column) =>
        reader.IsDBNull(column) ? null : JsonSerializer.Deserialize<Specifications>(reader.GetString(column));

    private static Vector DeserializeVector(string json) => new(JsonSerializer.Deserialize<double[]>(json) ?? []);

    private static string SerializeVector(Vector vector) => JsonSerializer.Serialize(vector.ToArray());
}
