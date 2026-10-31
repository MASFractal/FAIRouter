using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Core.Models.Common.Requests;
using AI.LLM.Core.Models.Common.Responses;
using FAI.Router.LLM;
using FAI.Router.RoutedElements;
using FAI.Router.Services;

namespace FAI.Router;

/// <summary>
/// Счет хода: обращения к моделям по порядку. Последняя попытка исполнителя дала ответ, предыдущие
/// отказали. Устроен так же, как spending.py в версии на Python.
/// </summary>
/// <remarks>
/// Обращения идут из разных мест и параллельно, а клиент судьи общий у всех ходов, поэтому счет живет
/// в контексте выполнения хода: заменам распознавателя и судьи у вызывающего ничего передавать не нужно.
/// </remarks>
internal sealed class Bill
{
    // Объем обращений к судье в токенах без текста задачи и ответа. Вход посчитан по длине промптов и
    // схем, выход по длине типичного вердикта на короткую задачу; величины не измерялись
    private const double RecognitionInput = 3300, RecognitionOutput = 200;
    private const double StyleInput = 1700, StyleOutput = 50;
    private const double ReviewInput = 1800, ReviewOutput = 150;

    // Сколько токенов ответа видит судья: AnswerChars судьи содержания при трех символах на токен
    private const double VisibleAnswerTokens = ContentJudge.AnswerChars / InputFeaturesService.EST_SYMBOL_PER_TOKEN;

    private static readonly AsyncLocal<Bill?> Active = new();

    private readonly Lock _gate = new();
    private readonly List<Call> _attempts = [];
    private readonly List<Call> _served = [];
    private readonly List<Call> _repeated = [];
    private Usage? _reported;

    /// <summary>Счет текущего хода; пусто вне хода</summary>
    public static Bill? Current => Active.Value;

    /// <summary>
    /// Расход следующей попытки исполнителя, когда поставщик его сообщил
    /// </summary>
    /// <param name="usage">Расход из ответа поставщика</param>
    public void Report(Usage? usage)
    {
        lock (_gate)
            _reported = usage;
    }

    /// <summary>
    /// Попытка исполнителя. Исполнитель, вернувший только текст, токенов не сообщает: они оцениваются по
    /// объему диалога и длине ответа. Поправка цены кандидата сюда не входит: она оценивает рассуждения
    /// заранее, а токены рассуждений уже есть в ответе поставщика.
    /// </summary>
    /// <param name="candidate">Кто отвечал</param>
    /// <param name="text">Его ответ</param>
    /// <param name="inputTokens">Оценка объема диалога</param>
    public void Attempted(BaseRoutedElement candidate, string? text, double inputTokens)
    {
        Price? price = candidate.HasKnownPrice ? new Price(candidate.DPMTInp, candidate.DPMTOutp) : null;

        lock (_gate)
        {
            _attempts.Add(IsReported(_reported)
                ? new Call(price, _reported!.PromptTokens, _reported.CompletionTokens, false)
                : new Call(price, inputTokens, Tokens(text?.Length ?? 0), true));
            _reported = null;
        }
    }

    /// <summary>
    /// Обращение к судье. Токены не сообщены, тогда они оцениваются по длине сообщений, схемы и ответа.
    /// </summary>
    /// <param name="usage">Расход из ответа поставщика</param>
    /// <param name="messages">Сообщения</param>
    /// <param name="settings">Настройки обращения со схемой ответа</param>
    /// <param name="text">Ответ модели</param>
    /// <param name="repeat">Повтор обращения</param>
    public void Served(Usage? usage, IReadOnlyList<LLMMessage> messages, GenerateSettings settings, string? text, bool repeat)
    {
        int asked = messages.Sum(message => (message.Content as string)?.Length ?? 0)
            + (settings.ResponseFormat?.JsonSchema?.Schema.GetRawText().Length ?? 0);
        Call call = IsReported(usage)
            ? new Call(null, usage!.PromptTokens, usage.CompletionTokens, false)
            : new Call(null, Tokens(asked), Tokens(text?.Length ?? 0), true);

        lock (_gate)
            (repeat ? _repeated : _served).Add(call);
    }

    /// <summary>
    /// Итог хода
    /// </summary>
    /// <param name="judgePrice">Цена модели судьи; пусто, если неизвестна</param>
    public RoundCost Total(Price? judgePrice)
    {
        lock (_gate)
        {
            Call[] served = [.. _served.Select(call => call with { Price = judgePrice })];
            Call[] repeated = [.. _repeated.Select(call => call with { Price = judgePrice })];
            Call[] answer = [.. _attempts.TakeLast(1)];
            Call[] retries = [.. _attempts.SkipLast(1), .. repeated];
            Call[] everything = [.. answer, .. served, .. retries];

            return new RoundCost(Sum(answer), Sum(served), Sum(retries), Sum(everything), everything.Any(call => call.Estimated));
        }
    }

    /// <summary>Заводит счет на время хода: в асинхронном методе он виден ему и всему, что он вызывает</summary>
    public static Bill Open() => Active.Value = new Bill();

    /// <summary>
    /// Ожидаемый расход судьи на ход, для цены выбора. У всех кандидатов он один, поэтому порядок цен не
    /// меняет, а меняет их отношение: на коротком запросе модель в двадцать раз дешевле дает те же
    /// несколько процентов итога, и платить за это качеством незачем. Распознавание к моменту выбора уже
    /// оплачено, но в итог задачи входит, а цена выбора оценивает именно его.
    /// </summary>
    /// <param name="taskTokens">Объем текста задачи</param>
    /// <param name="answerTokens">Ожидаемый объем ответа</param>
    /// <param name="price">Цена модели судьи; пусто, тогда ноль</param>
    /// <param name="measured">Оценивается ли ответ; иначе судья зовется только для распознавания</param>
    public static double ExpectedJudgeCost(double taskTokens, double answerTokens, Price? price, bool measured)
    {
        if (price is null)
            return 0;

        double visible = Math.Min(answerTokens, VisibleAnswerTokens);
        double input = RecognitionInput + taskTokens + (measured ? StyleInput + visible + ReviewInput + taskTokens + visible : 0);
        double output = RecognitionOutput + (measured ? StyleOutput + ReviewOutput : 0);

        return new Call(price, input, output, true).Amount!.Value;
    }

    private static bool IsReported(Usage? usage) => usage is { PromptTokens: > 0 } or { CompletionTokens: > 0 };

    private static double Tokens(int symbols) => symbols / InputFeaturesService.EST_SYMBOL_PER_TOKEN;

    private static double? Sum(Call[] calls) => calls.Any(call => call.Amount is null) ? null : calls.Sum(call => call.Amount);

    private readonly record struct Call(Price? Price, double PromptTokens, double CompletionTokens, bool Estimated)
    {
        public double? Amount => Price is null ? null : (Price.Input * PromptTokens + Price.Output * CompletionTokens) * 1e-6;
    }
}
