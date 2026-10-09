using System.Globalization;
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
/// По каждому смысловому пункту и каждому ограничению заказа судья отвечает отдельно, с номером
/// пункта, а уровень экспертности ответа называет по шкале экспертности заказа: критик сверяет с
/// заданием каждую из этих величин. Факты проверяются так же, как в замере фактологии: из ответа
/// выписываются атомарные проверяемые утверждения, и у каждого своя вероятность истинности. Без
/// проверки по вебу эту вероятность ставит сама модель-судья, то есть она сама себе фактчекер.
/// Хост с веб-поиском передает проверку делегатом, и тогда вероятность дает он. Неполный ответ
/// судьи (нет оценки хотя бы одного критерия) это сбой судьи, а не единица: оценки нет, и
/// автоотзыв по ходу не пишется. Задание и ответ идут судье внутри меток со случайным именем
/// (<see cref="PromptData"/>), указания внутри них судья не исполняет.
/// </remarks>
public class ContentJudge
{
    /// <summary>Сколько утверждений проверяется: больше дорого, меньше не хватает для средней</summary>
    public const int MaxClaims = 12;

    /// <summary>Сколько знаков ответа судья видит всегда; длинный заказ поднимает предел до <see cref="MaxAnswerChars"/></summary>
    internal const int AnswerChars = 24_000;

    /// <summary>Верхний предел показанного судье ответа: больше удорожает суд, а судья хуже держит длинный текст</summary>
    private const int MaxAnswerChars = 48_000;

    private static readonly string SystemPrompt =
        "Ты строгий эксперт-приемщик. Оцени СОДЕРЖАНИЕ ответа на задание, а не оформление: объем, "
        + $"число разделов и таблиц проверяет код. Выпиши до {MaxClaims} атомарных проверяемых утверждений "
        + "ответа (даты, числа, имена, нормы, характеристики) и для каждого вероятность, что оно "
        + "верно; мнения, оценки и вымысел не выписывай. По каждому смысловому пункту задания, с его "
        + "номером, оцени, насколько он раскрыт; по каждому ограничению, с его номером, соблюдено ли "
        + "оно. Уровень экспертности ответа оцени по той же шкале, что и экспертность задания. Затем "
        + "оцени критерии от 0 до 1 по опорным точкам. Длина и многословие сами по себе не достоинство: "
        + "повторы, вода и лишнее снижают пригодность для дела, короткий точный ответ не хуже длинного. "
        + "Если ответ обрезан для проверки, не считай упущенным пункт или ограничение, которые могли "
        + "оказаться в отрезанной части: оценивай показанное. В issues перечисли конкретные замечания по "
        + "содержанию: что именно неверно или упущено и где. Ответ хорош, значит issues пусто.";

    private static readonly string SchemaJson = BuildSchema();

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    private readonly LLMBase? _llm;
    private readonly Func<string, CancellationToken, Task<double?>>? _verify;

    /// <summary>
    /// Бюджет времени на суд, все попытки вместе: у движка таймаут 18 минут на попытку, и суд мог
    /// держать ход дольше, чем шел сам ответ
    /// </summary>
    public TimeSpan Budget { get; init; } = TimeSpan.FromMinutes(3);

    /// <summary>Модель судьи, если клиент знает свою модель; пусто, если не знает</summary>
    public string? Model => (_llm ?? (Settings.HasLLM ? Settings.LLM : null)) is OpenAiCompatibleLlm client ? client.Model : null;

    /// <summary>
    /// Судья содержания
    /// </summary>
    /// <param name="llm">
    /// Клиент модели; не задан, тогда берется общий Settings.LLM. Своя модель судьи задается здесь:
    /// модель, которая судит сама себя (судья среди кандидатов), завышает себе оценки.
    /// </param>
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
    /// <exception cref="InvalidDataException">Судья дважды ответил неполно или не по схеме</exception>
    /// <exception cref="TimeoutException">Судья не уложился в <see cref="Budget"/></exception>
    public async Task<ContentReview> ReviewAsync(
        string task, Specifications requested, string answer, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(answer))
            throw new ArgumentException("Ответ не может быть пустым.", nameof(answer));

        string tag = PromptData.NewTag();
        List<LLMMessage> messages =
        [
            new LLMMessage(LLMMessage.SystemRole, SystemPrompt + " " + PromptData.Rule(tag)),
            new LLMMessage(LLMMessage.UserRole, UserMessage(tag, task, requested, answer))
        ];

        GenerateSettings settings = JsonCall.Settings("content_review", SchemaJson);
        Verdict verdict = await JsonCall.AskAsync(_llm ?? Settings.LLM, messages, settings, Read, Budget, cancellationToken).ConfigureAwait(false);
        FactClaim[] claims = [.. ClaimsOf(verdict)];

        return Build(requested, verdict, _verify is null ? claims : await VerifyAsync(claims, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>
    /// Оценка по готовому ответу модели-судьи, без проверки утверждений по вебу. Нужна хосту,
    /// который хранит вердикты, и тестам: они проверяют сборку оценки без обращения к модели.
    /// </summary>
    /// <param name="requested">Распознанное задание</param>
    /// <param name="json">Ответ модели-судьи по схеме; ограда ```json и текст вокруг допускаются</param>
    /// <exception cref="InvalidDataException">Ответ неполон: нет оценки хотя бы одного критерия</exception>
    public static ContentReview FromJson(Specifications requested, string json)
    {
        Verdict verdict = JsonCall.Parse(json, Read)
            ?? throw new InvalidDataException("Ответ судьи неполон или не по схеме: оценки нет.");

        return Build(requested, verdict, [.. ClaimsOf(verdict)]);
    }

    /// <summary>
    /// Собирает оценку по ответу модели. Пункты и ограничения сопоставляются по номеру; номеров нет,
    /// а число совпало, тогда по порядку. Пункт без оценки получает общую полноту, ограничение без
    /// оценки считается соблюденным, если общая оценка выполнения указаний не ниже половины. Критерий,
    /// который к задаче не относится, остается пустым.
    /// </summary>
    internal static ContentReview Build(Specifications requested, Verdict verdict, IReadOnlyList<FactClaim> claims)
    {
        string[] points = [.. requested.RequiredPoints.Take(Specifications.MaxItems)];
        string[] constraints = [.. requested.Constraints.Take(Specifications.MaxItems)];
        double overallCompleteness = Clamp(verdict.Completeness!.Value);
        double overallInstruction = Clamp(verdict.InstructionFollowing!.Value);

        double?[] covered = Match(points.Length, verdict.Points ?? [], item => item.Index, item => item.Coverage);
        bool?[] met = Match(constraints.Length, verdict.Constraints ?? [], item => item.Index, item => item.Met);

        List<PointCoverage> coverage = [.. points.Select((point, i) => new PointCoverage(point, Clamp(covered[i] ?? overallCompleteness)))];
        List<ConstraintCheck> checks = [.. constraints.Select((constraint, i) => new ConstraintCheck(constraint, met[i] ?? overallInstruction >= 0.5))];

        double completeness = coverage.Count > 0 ? coverage.Average(item => item.Coverage) : overallCompleteness;
        double? instruction = checks.Count == 0 ? null : checks.Count(item => item.Met) / (double)checks.Count;

        return new(
            [
                new(ContentReview.Factuality, ContentReview.FactualityOf(claims)),
                new(ContentReview.Completeness, completeness),
                new(ContentReview.InstructionFollowing, instruction),
                new(ContentReview.Reasoning, Clamp(verdict.Reasoning!.Value)),
                new(ContentReview.Expertise, Clamp(verdict.Expertise!.Value)),
                new(ContentReview.StructureContent, Clamp(verdict.StructureContent!.Value)),
                new(ContentReview.SourceQuality, requested.HasReferences ? Clamp(verdict.SourceQuality!.Value) : null),
                new(ContentReview.FitForPurpose, Clamp(verdict.FitForPurpose!.Value)),
            ],
            claims,
            [.. (verdict.Issues ?? []).Where(issue => !string.IsNullOrWhiteSpace(issue))],
            coverage,
            checks,
            verdict.ExpertLevel is { } level && double.IsFinite(level) ? Clamp(level) : null);
    }

    // Проверка утверждений делегатом хоста разом. Сбой или нечисловой ответ на одном утверждении
    // оставляет ему оценку судьи, а не теряет весь разбор
    private async Task<FactClaim[]> VerifyAsync(FactClaim[] claims, CancellationToken cancellationToken)
    {
        double?[] checks = await Task.WhenAll(claims.Select(claim => CheckAsync(claim.Text, cancellationToken))).ConfigureAwait(false);

        return [.. claims.Select((claim, i) => checks[i] is { } truth ? claim with { Truth = Clamp(truth) } : claim)];
    }

    private async Task<double?> CheckAsync(string claim, CancellationToken cancellationToken)
    {
        try
        {
            double? truth = await _verify!(claim, cancellationToken).ConfigureAwait(false);

            return truth is { } value && double.IsFinite(value) ? value : null;
        }
        catch (Exception) when (!cancellationToken.IsCancellationRequested)
        {
            return null;
        }
    }

    private static double Clamp(double value) => Math.Clamp(value, 0, 1);

    // Вердикт без оценки хотя бы одного критерия или с нечисловой оценкой негоден: пропуск это не
    // единица и не ноль, а сбой судьи
    private static Verdict? Read(string json)
    {
        Verdict? verdict = JsonSerializer.Deserialize<Verdict>(json, JsonOptions);
        double?[] required =
        [
            verdict?.Completeness, verdict?.InstructionFollowing, verdict?.Reasoning, verdict?.Expertise,
            verdict?.StructureContent, verdict?.SourceQuality, verdict?.FitForPurpose
        ];

        return required.All(score => score is { } value && double.IsFinite(value)) ? verdict : null;
    }

    // Утверждения без вероятности или с нечисловой вероятностью отбрасываются: ноль по умолчанию
    // объявлял бы их ложными
    private static IEnumerable<FactClaim> ClaimsOf(Verdict verdict) =>
        (verdict.Claims ?? [])
            .Where(item => !string.IsNullOrWhiteSpace(item.Text) && item.Truth is { } truth && double.IsFinite(truth))
            .Take(MaxClaims)
            .Select(item => new FactClaim(item.Text!.Trim(), Clamp(item.Truth!.Value)));

    // Оценки судьи по номерам пунктов заказа. Номера есть у всех, тогда по номерам; номеров нет, а
    // число совпало, тогда по порядку (прежний вид ответа); иначе сопоставить нельзя, и пункт
    // получает общую оценку
    private static T?[] Match<TItem, T>(int count, IReadOnlyList<TItem> judged, Func<TItem, int?> number, Func<TItem, T?> value)
        where T : struct
    {
        T?[] matched = new T?[count];
        bool numbered = judged.Count > 0 && judged.All(item => number(item) is not null);

        for (int i = 0; i < judged.Count; i++)
        {
            int slot = numbered ? number(judged[i])!.Value - 1 : judged.Count == count ? i : -1;

            if (slot >= 0 && slot < count && matched[slot] is null)
                matched[slot] = value(judged[i]) is { } item && (item is not double score || double.IsFinite(score)) ? item : null;
        }

        return matched;
    }

    private static string UserMessage(string tag, string task, Specifications requested, string answer)
    {
        string points = Numbered(requested.RequiredPoints, "не выделены");
        string constraints = Numbered(requested.Constraints, "нет");
        int limit = Math.Clamp((int)Math.Min(requested.SymbolLength * 1.25, MaxAnswerChars), AnswerChars, MaxAnswerChars);

        return $"{PromptData.Wrap(tag, "задание", task)}\n\n"
            + $"{PromptData.Wrap(tag, "смысловые пункты", points)}\n\n"
            + $"{PromptData.Wrap(tag, "ограничения", constraints)}\n\n"
            + $"ЭКСПЕРТНОСТЬ ЗАДАНИЯ: {requested.ExpertLevel.ToString("0.00", CultureInfo.InvariantCulture)}\n\n"
            + $"НУЖНЫ ИСТОЧНИКИ: {(requested.HasReferences ? "да" : "нет")}\n\n"
            + PromptData.Wrap(tag, "ответ", PromptData.Clip(answer, limit));
    }

    private static string Numbered(IReadOnlyList<string> items, string empty) =>
        items.Count == 0 ? empty : string.Join("\n", items.Take(Specifications.MaxItems).Select((item, i) => $"{i + 1}. {item}"));

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
                    description = $"До {MaxClaims} атомарных проверяемых утверждений ответа",
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
                            index = new { type = "integer", description = "Номер пункта в задании" },
                            point = new { type = "string", description = "Пункт задания" },
                            coverage = new { type = "number", minimum = 0, maximum = 1, description = "Насколько раскрыт" }
                        },
                        required = new[] { "index", "point", "coverage" },
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
                            index = new { type = "integer", description = "Номер ограничения в задании" },
                            constraint = new { type = "string", description = "Ограничение задания" },
                            met = new { type = "boolean", description = "Соблюдено ли" }
                        },
                        required = new[] { "index", "constraint", "met" },
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

    /// <summary>Ответ модели по схеме; пропущенное поле остается пустым, а не получает оценку</summary>
    internal sealed class Verdict
    {
        public List<VerdictClaim>? Claims { get; set; }
        public List<VerdictPoint>? Points { get; set; }
        public List<VerdictConstraint>? Constraints { get; set; }
        public double? ExpertLevel { get; set; }
        public double? Completeness { get; set; }
        public double? InstructionFollowing { get; set; }
        public double? Reasoning { get; set; }
        public double? Expertise { get; set; }
        public double? StructureContent { get; set; }
        public double? SourceQuality { get; set; }
        public double? FitForPurpose { get; set; }
        public List<string>? Issues { get; set; }
    }

    /// <summary>Утверждение в ответе модели</summary>
    internal sealed class VerdictClaim
    {
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("truth")]
        public double? Truth { get; set; }
    }

    /// <summary>Раскрытие пункта в ответе модели</summary>
    internal sealed class VerdictPoint
    {
        [JsonPropertyName("index")]
        public int? Index { get; set; }

        [JsonPropertyName("point")]
        public string? Point { get; set; }

        [JsonPropertyName("coverage")]
        public double? Coverage { get; set; }
    }

    /// <summary>Соблюдение ограничения в ответе модели</summary>
    internal sealed class VerdictConstraint
    {
        [JsonPropertyName("index")]
        public int? Index { get; set; }

        [JsonPropertyName("constraint")]
        public string? Constraint { get; set; }

        [JsonPropertyName("met")]
        public bool? Met { get; set; }
    }
}

/// <summary>
/// Опорные точки критериев содержания. Без них модель ставит оценки наугад, как было со шкалой
/// терминологии; текст общий с версией на Python.
/// </summary>
internal static class ContentCriteriaDescriptions
{
    public const string Points =
        "По каждому смысловому пункту задания с его номером: насколько он раскрыт, 0-1. 1 - "
        + "раскрыт по сути; 0.5 - упомянут без раскрытия; 0 - отсутствует или раскрыт неверно. "
        + "Пусто, если пунктов нет.";

    public const string Constraints =
        "По каждому ограничению задания с его номером: соблюдено ли оно. Пусто, если ограничений нет.";

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
        + "1 - уровень опытного профессионала. Объем глубиной не считается.";

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
