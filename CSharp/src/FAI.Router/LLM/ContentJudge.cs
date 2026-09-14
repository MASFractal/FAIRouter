using System.Text.Json;
using System.Text.Json.Serialization;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Core.Models.Common.Requests;
using AI.LLM.Services.LLM;
using FAI.Router.JudgeLogic;

namespace FAI.Router.LLM;

/// <summary>
/// Судья содержания: одно обращение к модели по строгой схеме. Оценивает то, что сверка формы не
/// видит: верность фактов, полноту по сути, выполнение указаний, рассуждения, глубину,
/// наполненность структуры, источники и пригодность для дела.
/// </summary>
/// <remarks>
/// По каждому смысловому пункту и каждому ограничению заказа судья отвечает отдельно, а уровень
/// экспертности ответа называет по шкале экспертности заказа: критик сверяет с заданием каждую из
/// этих величин. Факты проверяются так же, как в рейтинге фактологии арены: из ответа
/// выписываются атомарные проверяемые утверждения, и у каждого своя вероятность истинности. Без
/// проверки по вебу эту вероятность ставит сама модель-судья, то есть она сама себе фактчекер.
/// Хост с веб-поиском передает проверку делегатом, и тогда вероятность дает он.
/// </remarks>
public class ContentJudge
{
    /// <summary>Сколько утверждений проверяется: больше дорого, меньше не хватает для средней</summary>
    public const int MaxClaims = 12;

    private const int AnswerChars = 24_000;

    private const string SystemPrompt =
        "Ты строгий эксперт-приемщик. Оцени СОДЕРЖАНИЕ ответа на задание, а не оформление: объем, "
        + "число разделов и таблиц проверяет код. Выпиши до 12 атомарных проверяемых утверждений "
        + "ответа (даты, числа, имена, нормы, характеристики) и для каждого вероятность, что оно "
        + "верно; мнения, оценки и вымысел не выписывай. По каждому смысловому пункту задания, в том "
        + "же порядке, оцени, насколько он раскрыт; по каждому ограничению, в том же порядке, "
        + "соблюдено ли оно. Уровень экспертности ответа оцени по той же шкале, что и экспертность "
        + "задания. Затем оцени критерии от 0 до 1 по опорным точкам. В issues перечисли конкретные "
        + "замечания по содержанию: что именно неверно или упущено и где. Ответ хорош, значит issues пусто.";

    private static readonly string SchemaJson = BuildSchema();

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    private readonly LLMBase? _llm;
    private readonly Func<string, CancellationToken, Task<double?>>? _verify;

    /// <summary>
    /// Судья содержания
    /// </summary>
    /// <param name="llm">Клиент модели; не задан, тогда берется общий Settings.LLM</param>
    /// <param name="verify">
    /// Проверка утверждения по внешнему источнику (веб-поиск хоста): вероятность истинности или
    /// пусто, если проверить не удалось. Не задана, тогда вероятность ставит модель-судья.
    /// </param>
    public ContentJudge(LLMBase? llm = null, Func<string, CancellationToken, Task<double?>>? verify = null)
    {
        _llm = llm;
        _verify = verify;
    }

    /// <summary>
    /// Оценивает содержание ответа на задание
    /// </summary>
    /// <param name="task">Текст задания</param>
    /// <param name="requested">Распознанное задание: смысловые пункты, ограничения, нужны ли источники</param>
    /// <param name="answer">Ответ исполнителя</param>
    /// <param name="cancellationToken">Токен отмены</param>
    public async Task<ContentReview> ReviewAsync(
        string task, Specifications requested, string answer, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(answer))
            throw new ArgumentException("Ответ не может быть пустым.", nameof(answer));

        GenerateSettings settings = new(temperature: 0)
        {
            ResponseFormat = ResponseFormat.CreateJsonSchema("content_review", SchemaJson)
        };

        List<LLMMessage> messages =
        [
            new LLMMessage(LLMMessage.SystemRole, SystemPrompt),
            new LLMMessage(LLMMessage.UserRole, UserMessage(task, requested, answer))
        ];

        string json = await (_llm ?? Settings.LLM).SendToLLM(messages, settings, cancellationToken).ConfigureAwait(false);
        Verdict verdict = Read(json);

        List<FactClaim> claims = [];

        foreach (FactClaim claim in ClaimsOf(verdict))
        {
            double? checkedTruth = _verify is null ? null : await _verify(claim.Text, cancellationToken).ConfigureAwait(false);
            claims.Add(claim with { Truth = Math.Clamp(checkedTruth ?? claim.Truth, 0, 1) });
        }

        return Build(requested, verdict, claims);
    }

    /// <summary>
    /// Оценка по готовому ответу модели-судьи, без проверки утверждений по вебу. Нужна хосту,
    /// который хранит вердикты, и тестам: они проверяют сборку оценки без обращения к модели.
    /// </summary>
    /// <param name="requested">Распознанное задание</param>
    /// <param name="json">Ответ модели-судьи по схеме</param>
    public static ContentReview FromJson(Specifications requested, string json)
    {
        Verdict verdict = Read(json);

        return Build(requested, verdict, [.. ClaimsOf(verdict)]);
    }

    /// <summary>
    /// Собирает оценку по ответу модели. Пункты и ограничения берутся в порядке заказа: пропущенный
    /// судьей пункт получает общую полноту, пропущенное ограничение считается соблюденным, если
    /// общая оценка выполнения указаний не ниже половины. Критерий, который к задаче не относится,
    /// остается пустым.
    /// </summary>
    internal static ContentReview Build(Specifications requested, Verdict verdict, IReadOnlyList<FactClaim> claims)
    {
        List<VerdictPoint> points = verdict.Points ?? [];
        List<VerdictConstraint> constraints = verdict.Constraints ?? [];

        List<PointCoverage> coverage =
        [
            .. requested.RequiredPoints.Select((point, i) =>
                new PointCoverage(point, Clamp(i < points.Count ? points[i].Coverage : verdict.Completeness)))
        ];
        List<ConstraintCheck> checks =
        [
            .. requested.Constraints.Select((constraint, i) =>
                new ConstraintCheck(constraint, i < constraints.Count ? constraints[i].Met : verdict.InstructionFollowing >= 0.5))
        ];

        double completeness = coverage.Count > 0 ? coverage.Average(item => item.Coverage) : Clamp(verdict.Completeness);
        double? instruction = checks.Count == 0 ? null : checks.Count(item => item.Met) / (double)checks.Count;

        return new(
            [
                new(ContentReview.Factuality, ContentReview.FactualityOf(claims)),
                new(ContentReview.Completeness, completeness),
                new(ContentReview.InstructionFollowing, instruction),
                new(ContentReview.Reasoning, Clamp(verdict.Reasoning)),
                new(ContentReview.Expertise, Clamp(verdict.Expertise)),
                new(ContentReview.StructureContent, Clamp(verdict.StructureContent)),
                new(ContentReview.SourceQuality, requested.HasReferences ? Clamp(verdict.SourceQuality) : null),
                new(ContentReview.FitForPurpose, Clamp(verdict.FitForPurpose)),
            ],
            claims,
            [.. (verdict.Issues ?? []).Where(issue => !string.IsNullOrWhiteSpace(issue))],
            coverage,
            checks,
            verdict.ExpertLevel is { } level ? Clamp(level) : null);
    }

    private static double Clamp(double value) => Math.Clamp(value, 0, 1);

    private static Verdict Read(string json) => JsonSerializer.Deserialize<Verdict>(json, JsonOptions) ?? new Verdict();

    private static IEnumerable<FactClaim> ClaimsOf(Verdict verdict) =>
        (verdict.Claims ?? [])
            .Where(item => !string.IsNullOrWhiteSpace(item.Text))
            .Take(MaxClaims)
            .Select(item => new FactClaim(item.Text!.Trim(), Clamp(item.Truth)));

    private static string UserMessage(string task, Specifications requested, string answer)
    {
        string points = requested.RequiredPoints.Count == 0 ? "не выделены" : Numbered(requested.RequiredPoints);
        string constraints = requested.Constraints.Count == 0 ? "нет" : Numbered(requested.Constraints);
        string clipped = answer.Length <= AnswerChars ? answer : answer[..AnswerChars] + "\n[…ответ обрезан для судьи]";

        return $"ЗАДАНИЕ:\n{task}\n\nСМЫСЛОВЫЕ ПУНКТЫ: {points}\n\nОГРАНИЧЕНИЯ: {constraints}\n\n"
            + $"ЭКСПЕРТНОСТЬ ЗАДАНИЯ: {requested.ExpertLevel:0.00}\n\n"
            + $"НУЖНЫ ИСТОЧНИКИ: {(requested.HasReferences ? "да" : "нет")}\n\nОТВЕТ:\n{clipped}";
    }

    private static string Numbered(IReadOnlyList<string> items) =>
        "\n" + string.Join("\n", items.Select((item, i) => $"{i + 1}. {item}"));

    private static string BuildSchema()
    {
        var schema = new
        {
            type = "object",
            properties = new
            {
                claims = new
                {
                    type = "array",
                    description = "До 12 атомарных проверяемых утверждений ответа",
                    items = new
                    {
                        type = "object",
                        properties = new
                        {
                            text = new { type = "string", description = "Утверждение одной фразой" },
                            truth = new { type = "number", minimum = 0, maximum = 1, description = "Вероятность, что утверждение верно" }
                        },
                        required = new[] { "text", "truth" },
                        additionalProperties = false
                    }
                },
                points = new
                {
                    type = "array",
                    description = ContentCriteriaDescriptions.Points,
                    items = new
                    {
                        type = "object",
                        properties = new
                        {
                            point = new { type = "string", description = "Пункт задания" },
                            coverage = new { type = "number", minimum = 0, maximum = 1, description = "Насколько раскрыт" }
                        },
                        required = new[] { "point", "coverage" },
                        additionalProperties = false
                    }
                },
                constraints = new
                {
                    type = "array",
                    description = ContentCriteriaDescriptions.Constraints,
                    items = new
                    {
                        type = "object",
                        properties = new
                        {
                            constraint = new { type = "string", description = "Ограничение задания" },
                            met = new { type = "boolean", description = "Соблюдено ли" }
                        },
                        required = new[] { "constraint", "met" },
                        additionalProperties = false
                    }
                },
                expertLevel = Criterion(ContentCriteriaDescriptions.ExpertLevel),
                completeness = Criterion(ContentCriteriaDescriptions.Completeness),
                instructionFollowing = Criterion(ContentCriteriaDescriptions.InstructionFollowing),
                reasoning = Criterion(ContentCriteriaDescriptions.Reasoning),
                expertise = Criterion(ContentCriteriaDescriptions.Expertise),
                structureContent = Criterion(ContentCriteriaDescriptions.StructureContent),
                sourceQuality = Criterion(ContentCriteriaDescriptions.SourceQuality),
                fitForPurpose = Criterion(ContentCriteriaDescriptions.FitForPurpose),
                issues = new { type = "array", items = new { type = "string" }, description = "Конкретные замечания по содержанию" }
            },
            required = new[]
            {
                "claims", "points", "constraints", "expertLevel", "completeness", "instructionFollowing",
                "reasoning", "expertise", "structureContent", "sourceQuality", "fitForPurpose", "issues"
            },
            additionalProperties = false
        };

        return JsonSerializer.Serialize(schema);
    }

    private static object Criterion(string description) => new { type = "number", minimum = 0, maximum = 1, description };

    /// <summary>Ответ модели по схеме</summary>
    internal sealed class Verdict
    {
        public List<VerdictClaim>? Claims { get; set; }
        public List<VerdictPoint>? Points { get; set; }
        public List<VerdictConstraint>? Constraints { get; set; }
        public double? ExpertLevel { get; set; }
        public double Completeness { get; set; } = 1;
        public double InstructionFollowing { get; set; } = 1;
        public double Reasoning { get; set; } = 1;
        public double Expertise { get; set; } = 1;
        public double StructureContent { get; set; } = 1;
        public double SourceQuality { get; set; } = 1;
        public double FitForPurpose { get; set; } = 1;
        public List<string>? Issues { get; set; }
    }

    /// <summary>Утверждение в ответе модели</summary>
    internal sealed class VerdictClaim
    {
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("truth")]
        public double Truth { get; set; }
    }

    /// <summary>Раскрытие пункта в ответе модели</summary>
    internal sealed class VerdictPoint
    {
        [JsonPropertyName("point")]
        public string? Point { get; set; }

        [JsonPropertyName("coverage")]
        public double Coverage { get; set; }
    }

    /// <summary>Соблюдение ограничения в ответе модели</summary>
    internal sealed class VerdictConstraint
    {
        [JsonPropertyName("constraint")]
        public string? Constraint { get; set; }

        [JsonPropertyName("met")]
        public bool Met { get; set; }
    }
}

/// <summary>
/// Опорные точки критериев содержания. Без них модель ставит оценки наугад, как было со шкалой
/// терминологии; текст общий с версией на Python.
/// </summary>
internal static class ContentCriteriaDescriptions
{
    public const string Points =
        "По каждому смысловому пункту задания в том же порядке: насколько он раскрыт, 0-1. 1 - "
        + "раскрыт по сути; 0.5 - упомянут без раскрытия; 0 - отсутствует или раскрыт неверно. "
        + "Пусто, если пунктов нет.";

    public const string Constraints =
        "По каждому ограничению задания в том же порядке: соблюдено ли оно. Пусто, если ограничений нет.";

    public const string ExpertLevel =
        "Уровень экспертности самого ответа, 0-1, по той же шкале, что экспертность задания: 0.1 - "
        + "бытовой уровень; 0.4 - грамотный пользователь; 0.7 - специалист; 0.9 - эксперт.";

    public const string Completeness =
        "Раскрыты ли смысловые пункты задания по сути, 0-1. 1 - каждый пункт раскрыт содержательно; "
        + "0.6 - часть пунктов упомянута без раскрытия; 0.3 - раскрыта меньшая часть; 0 - ответ не о том.";

    public const string InstructionFollowing =
        "Доля выполненных явных ограничений задания, 0-1. Если ограничений нет, 1.";

    public const string Reasoning =
        "Верность рассуждений и расчетов, 0-1. 1 - выводы следуют из данных, числа сходятся; 0.5 - "
        + "есть недоказанные выводы или мелкие ошибки в расчетах; 0 - выводы противоречат данным или "
        + "расчеты неверны. Если рассуждений и расчетов нет, оцени логику изложения.";

    public const string Expertise =
        "Глубина, которой ждет специалист области, 0-1. 0.2 - общие слова, подошедшие бы к любой "
        + "задаче; 0.5 - грамотно, но поверхностно; 0.8 - конкретика, термины и нюансы по делу; "
        + "1 - уровень опытного профессионала.";

    public const string StructureContent =
        "Содержательность структуры, 0-1: таблицы, списки и разделы наполнены данными по делу. "
        + "1 - каждая строка несет содержание; 0.5 - часть строк пустые, повторяются или общие; "
        + "0 - структура есть, а содержания в ней нет или оно выдумано.";

    public const string SourceQuality =
        "Качество источников, 0-1: источники правдоподобно существуют, относятся к делу и "
        + "подтверждают утверждения. 1 - все такие; 0.5 - часть не по делу или непроверяема; 0 - "
        + "источники выдуманы или их нет. Если источники не нужны, 1.";

    public const string FitForPurpose =
        "Пригодность для дела, 0-1: можно ли отдать результат заказчику как есть. 1 - как есть; "
        + "0.7 - после мелкой правки; 0.4 - нужна существенная переделка; 0 - непригоден.";
}
