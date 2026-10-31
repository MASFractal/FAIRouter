using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using FAI.Router.Catalog;
using FAI.Router.Enums;
using FAI.Router.LLM;
using FAI.Router.RoutedElements;
using static FAI.Router.Tests.Scene;

namespace FAI.Router.Tests;

/// <summary>Счет хода: ответ отдельно, судья отдельно, повторы отдельно. Те же случаи, что в test_spending.py</summary>
public class SpendingTests
{
    private static readonly Price JudgePrice = new(10, 100);

    // Одно обращение к судье: 100 токенов входа и 10 выхода по его цене
    private const double JudgeCall = (10 * 100 + 100 * 10) * 1e-6;

    private static readonly RouteWeights ByQuality = new(1, 0, 0, 0);

    private static List<BaseRoutedElement> Candidates()
    {
        var features = Features();
        return [Element(features, "leader", 0.9, 1), Element(features, "spare", 0.5, 3)];
    }

    private static FaiRouter Router(MeteredProvider provider, Func<BaseRoutedElement, string> answer, List<BaseRoutedElement>? elements = null) =>
        new(elements ?? Candidates(), (candidate, _, _) => Task.FromResult(answer(candidate)), weights: ByQuality, llm: provider.Client) { JudgePrice = JudgePrice };

    /// <summary>
    /// Путь фабрики целиком: расход исполнителя и судьи приходит из потока поставщика, а итог задачи
    /// это ответ плюс три обращения к судье, в том числе параллельные
    /// </summary>
    [Fact]
    public async Task Factory_turn_separates_answer_from_judge()
    {
        using MeteredProvider provider = new();
        Dictionary<string, Price> prices = new() { ["a/model"] = new Price(1, 2), [FaiRouter.DefaultJudgeModel] = JudgePrice };

        FaiRouter router = await FaiRouter.FromOpenAiCompatibleAsync(provider.Url, "rtr_test", ["a/model"], prices: prices);
        RoundCost cost = (await router.AskAsync("Напиши обзор")).Cost!;

        Assert.Equal((1 * 1000 + 2 * 500) * 1e-6, cost.Answer!.Value, 12);
        Assert.Equal(3 * JudgeCall, cost.Judge!.Value, 12);
        Assert.Equal(0, cost.Retries);
        Assert.Equal(cost.Answer.Value + cost.Judge.Value, cost.Total!.Value, 12);
        Assert.False(cost.Estimated);
    }

    /// <summary>Цена судьи берется из prices, затем из каталога; нет нигде, тогда она неизвестна</summary>
    [Fact]
    public async Task Factory_takes_judge_price_from_prices_and_catalog()
    {
        Dictionary<string, Price> own = new() { ["a/model"] = new Price(1, 2) };
        Dictionary<string, Price> priced = new(own) { [FaiRouter.DefaultJudgeModel] = JudgePrice };
        List<ModelInfo> catalog = [new("b/model", "b", 1, 2, 0, 0, Capability.All), new(FaiRouter.DefaultJudgeModel, "j", 5, 6, 0, 0, Capability.All)];

        Assert.Equal(JudgePrice, (await FaiRouter.FromOpenAiCompatibleAsync("http://127.0.0.1:1/v1", "k", ["a/model"], prices: priced, measure: false)).JudgePrice);
        Assert.Null((await FaiRouter.FromOpenAiCompatibleAsync("http://127.0.0.1:1/v1", "k", ["a/model"], prices: own, measure: false)).JudgePrice);
        Assert.Equal(new Price(5, 6), (await FaiRouter.FromOpenAiCompatibleAsync("http://127.0.0.1:1/v1", "k", ["b/model"],
            catalogFetch: _ => Task.FromResult<IReadOnlyList<ModelInfo>>(catalog), measure: false)).JudgePrice);
    }

    /// <summary>Пустой ответ поставщик считает, поэтому он идет в повторы; об ошибке расхода никто не сообщил</summary>
    [Fact]
    public async Task Empty_answer_is_paid_and_failure_is_not()
    {
        using MeteredProvider provider = new();

        RoundCost empty = (await Router(provider, candidate => candidate.Name == "leader" ? "" : "ответ").AskAsync("x")).Cost!;
        RoundCost broken = (await Router(provider, candidate => candidate.Name == "leader" ? throw new IOException("обрыв связи") : "ответ").AskAsync("x")).Cost!;

        Assert.True(empty.Retries > 0);
        Assert.Equal(0, broken.Retries);
        Assert.Equal(empty.Answer, broken.Answer);
        Assert.Equal(empty.Answer!.Value + empty.Judge!.Value + empty.Retries!.Value, empty.Total!.Value, 12);
    }

    [Fact]
    public async Task Repeat_of_judge_goes_to_retries()
    {
        using MeteredProvider provider = new() { Cut = "content_review" };

        RoundCost cost = (await Router(provider, _ => "ответ").AskAsync("Напиши обзор")).Cost!;

        Assert.Equal(3 * JudgeCall, cost.Judge!.Value, 12);
        Assert.Equal(JudgeCall, cost.Retries!.Value, 12);
    }

    /// <summary>Без цены итога нет: он выглядел бы полным, а им не был. Ответ при этом посчитан</summary>
    [Fact]
    public async Task Unknown_price_leaves_total_unknown()
    {
        using MeteredProvider provider = new();
        FaiRouter unpricedJudge = Router(provider, _ => "ответ");
        unpricedJudge.JudgePrice = null;

        RoundCost cost = (await unpricedJudge.AskAsync("x")).Cost!;

        Assert.NotNull(cost.Answer);
        Assert.Null(cost.Judge);
        Assert.Null(cost.Total);

        List<BaseRoutedElement> elements = Candidates();
        elements[0].DPMTInp = elements[0].DPMTOutp = -1;
        cost = (await Router(provider, _ => "ответ", [elements[0]]).AskAsync("x")).Cost!;

        Assert.Null(cost.Answer);
        Assert.NotNull(cost.Judge);
        Assert.Null(cost.Total);
    }

    /// <summary>Исполнитель вернул только текст: токены оценены по его длине, и счет это отмечает</summary>
    [Fact]
    public async Task Tokens_are_estimated_when_nobody_reports_them()
    {
        using MeteredProvider provider = new();

        RoundCost cost = (await Router(provider, _ => new string('y', 600)).AskAsync(new string('x', 300))).Cost!;

        Assert.Equal((1 * 100 + 5 * 200) * 1e-6, cost.Answer!.Value, 12);
        Assert.True(cost.Estimated);
    }

    /// <summary>Распознавание платится всегда, стиль и суд только с замером; судья видит ответ не целиком</summary>
    [Fact]
    public void Expected_judge_cost_follows_task_and_answer()
    {
        Price price = new(1, 0);

        Assert.Equal(0, Bill.ExpectedJudgeCost(10, 100, null, true));
        Assert.Equal(3310 * 1e-6, Bill.ExpectedJudgeCost(10, 100, price, false), 12);
        Assert.Equal((3310 + 1800 + 1910) * 1e-6, Bill.ExpectedJudgeCost(10, 100, price, true), 12);
        Assert.Equal(Bill.ExpectedJudgeCost(10, 8000, price, true), Bill.ExpectedJudgeCost(10, 1e7, price, true), 12);
    }

    /// <summary>
    /// Экономный профиль больше не отдает качество за экономию, которой в итоге задачи нет: без цены
    /// судьи выбор как прежде, на коротком запросе разница ответов тонет в расходе судьи, а на длинном
    /// экономия настоящая
    /// </summary>
    [Theory]
    [InlineData(30, false, "cheap")]
    [InlineData(30, true, "strong")]
    [InlineData(30000, true, "cheap")]
    public async Task Judge_cost_flattens_price_gap_only_on_short_tasks(int symbols, bool priced, string winner)
    {
        using MeteredProvider provider = new() { Symbols = symbols };
        var features = Features();
        BaseRoutedElement cheap = Element(features, "cheap", 0.5, 0.04);
        BaseRoutedElement strong = Element(features, "strong", 0.9, 2);
        cheap.DPMTOutp = 0.08;
        strong.DPMTOutp = 10;

        FaiRouter router = new([cheap, strong], (_, _, _) => Task.FromResult("ответ"), measure: false,
            weights: new RouteWeights(0.3, 0.60, 0.10, 0), llm: provider.Client)
        { JudgePrice = priced ? new Price(0.15, 0.6) : null };

        Assert.Equal(winner, (await router.AskAsync("Какая столица у Австралии?")).Winner.Name);
    }

    /// <summary>Поставщик на локальном порту: отвечает потоком и в конце сообщает расход, как настоящий</summary>
    private sealed class MeteredProvider : IDisposable
    {
        private const string Verdict = """
            { "claims": [], "points": [], "constraints": [], "expertLevel": 0.7, "completeness": 0.9,
              "instructionFollowing": 1.0, "reasoning": 0.9, "expertise": 0.8, "structureContent": 0.8,
              "sourceQuality": 0.7, "fitForPurpose": 0.8, "issues": [] }
            """;

        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        private readonly ConcurrentDictionary<string, int> _seen = new();

        public MeteredProvider()
        {
            _listener.Start();
            _ = Task.Run(ServeAsync);
        }

        /// <summary>Первый ответ этой схемы обрезается по потолку</summary>
        public string? Cut { get; init; }

        /// <summary>Заказанный объем в распознанном задании</summary>
        public int Symbols { get; init; } = 1500;

        public string Url => $"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}/v1";

        public OpenAiCompatibleLlm Client => new(Url, "rtr_test", "judge/model");

        public void Dispose() => _listener.Stop();

        private async Task ServeAsync()
        {
            try
            {
                while (true)
                {
                    TcpClient client = await _listener.AcceptTcpClientAsync();
                    _ = Task.Run(() => AnswerAsync(client));
                }
            }
            catch (Exception error) when (error is SocketException or ObjectDisposedException)
            {
            }
        }

        private async Task AnswerAsync(TcpClient client)
        {
            using (client)
            {
                NetworkStream stream = client.GetStream();
                string request = await ReadRequestAsync(stream);
                string schema = new[] { "input_specifications", "style_assessment", "content_review" }.FirstOrDefault(request.Contains) ?? "";
                bool cut = schema == Cut && _seen.AddOrUpdate(schema, 1, (_, count) => count + 1) == 1;

                string content = schema switch
                {
                    "input_specifications" => $$"""{ "styleType": "Scientific", "symbolLength": {{Symbols}}, "language": "ru" }""",
                    "style_assessment" => """{ "styleType": "Scientific", "termDensity": 0.65, "formalityScore": 0.85 }""",
                    "content_review" => Verdict,
                    _ => "ответ",
                };
                var usage = schema == "" ? new { prompt_tokens = 1000, completion_tokens = 500 } : new { prompt_tokens = 100, completion_tokens = 10 };
                string text = JsonSerializer.Serialize(new { choices = new[] { new { index = 0, delta = new { content }, finish_reason = cut ? "length" : "stop" } } });
                string tail = JsonSerializer.Serialize(new { choices = Array.Empty<object>(), usage });
                string body = $"data: {text}\n\ndata: {tail}\n\ndata: [DONE]\n\n";
                string head = $"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {Encoding.UTF8.GetByteCount(body)}\r\nConnection: close\r\n\r\n";

                await stream.WriteAsync(Encoding.UTF8.GetBytes(head + body));
            }
        }

        private static async Task<string> ReadRequestAsync(NetworkStream stream)
        {
            byte[] buffer = new byte[64 * 1024];
            using MemoryStream received = new();

            while (true)
            {
                int read = await stream.ReadAsync(buffer);
                received.Write(buffer, 0, read);
                string text = Encoding.UTF8.GetString(received.ToArray());
                int split = text.IndexOf("\r\n\r\n", StringComparison.Ordinal);

                if (read == 0 || split < 0)
                {
                    if (read == 0)
                        return text;

                    continue;
                }

                string length = text[..split].Split("\r\n").First(line => line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase))["Content-Length:".Length..];

                if (received.Length >= Encoding.UTF8.GetByteCount(text[..(split + 4)]) + int.Parse(length))
                    return text;
            }
        }
    }
}
