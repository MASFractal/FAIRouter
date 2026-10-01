using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Services.LLM;
using FAI.Router;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.Persistence;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using FAI.Router.Services;
using FAI.Router.Training;

// Живой прогон: настоящие ходы настоящими моделями, оценка судьи, накопление и обучение
// на накопленном. Отличается от синтетических проверок тем, что ответы никто не подбирал.

string keyFile = Path.Combine(AppContext.BaseDirectory, "key.txt");
string apiKey = Environment.GetEnvironmentVariable("OPENROUTER_API_KEY")
    ?? (File.Exists(keyFile) ? File.ReadAllText(keyFile).Trim() : "");

if (string.IsNullOrWhiteSpace(apiKey))
{
    Console.WriteLine("Нет ключа: задайте OPENROUTER_API_KEY или положите key.txt рядом с программой.");
    return;
}

LLMBase Client(string model) => new LLMWithOpenRouterClient(new LLMOptions { ApiKey = apiKey, ModelName = model });

// Распознаванием задачи и разбором ответа занимается одна модель, по измерению она лучшая
Settings.LLM = Client("openai/gpt-4o-mini");

// Кандидаты: цены и скорость взяты из прайса OpenRouter
(BaseRoutedElement Element, LLMBase Llm)[] candidates =
[
    (new BaseRoutedElement { Name = "gemini-2.5-flash", TPS = 200, DPMTInp = 0.30, DPMTOutp = 2.50 }, Client("google/gemini-2.5-flash")),
    (new BaseRoutedElement { Name = "gpt-4.1-mini",     TPS = 90,  DPMTInp = 0.40, DPMTOutp = 1.60 }, Client("openai/gpt-4.1-mini")),
    (new BaseRoutedElement { Name = "claude-haiku-4.5", TPS = 60,  DPMTInp = 1.00, DPMTOutp = 5.00 }, Client("anthropic/claude-haiku-4.5"))
];

BaseRoutedElement[] elements = [.. candidates.Select(item => item.Element)];

string[] prompts =
[
    "Объясни восьмилетнему ребёнку простыми словами, что такое кластеризация. Уложись в 600 знаков.",
    "Напиши краткий научный обзор методов кластеризации на 1500 знаков, раздели на 3 раздела, добавь таблицу сравнения.",
    "Составь официальное уведомление подрядчику о сроках сдачи отчёта. 700 знаков, деловой стиль.",
    "Опиши алгоритм k-means для технической документации: 1000 знаков, с блоком кода на Python.",
    "Расскажи в разговорном стиле, зачем нужна кластеризация. 500 знаков, без формул.",
    "Подготовь научное описание спектральной кластеризации на 1200 знаков со ссылками на источники."
];

string dbPath = Path.Combine(AppContext.BaseDirectory, "live-session.db");
File.Delete(dbPath);

SqliteTraceStore traces = new(dbPath);
SqliteWeightsStore weights = new(dbPath);
SpecOutputService measurer = new();
Judge judge = new();

Console.WriteLine($"Кандидатов {elements.Length}, запросов {prompts.Length}");
Console.WriteLine();

foreach (string prompt in prompts)
{
    // 1. Ход роутинга: признаки задачи и выбор исполнителя
    Tracert trace = await Env.RouteAsync(prompt, elements);
    LLMBase winnerLlm = candidates.First(item => item.Element == trace.Winner).Llm;

    // 2. Победитель действительно отвечает
    string answer = await winnerLlm.SendToLLM([new LLMMessage(LLMMessage.UserRole, prompt)]);

    // 3. Замер факта и оценка судьи, причем оценка идет в саму трассировку
    Specifications requested = trace.RequestedSpec!;
    Specifications actual = await measurer.GetSpecificationsAsync(answer);
    double score = judge.Rate(trace, requested, actual);

    DiffSpec diff = Judge.Criticize(requested, actual);

    // 4. Ход попадает в накопитель, а отзыв ставится автоматически по разбору критика
    long roundId = traces.Append(trace, requested, actual, prompt);
    traces.SetFeedback(roundId, new Feedback
    {
        FType = FeedbackType.Auto,
        FeadbackScore = 1 - diff.TotalDeviation
    });

    Console.WriteLine($"[{roundId}] {trace.Winner.Name,-18} {requested.StyleType,-18} -> {actual.StyleType,-18} " +
        $"знаков {actual.SymbolLength,5} из {requested.SymbolLength,5}   судья {score:F3}   отзыв {1 - diff.TotalDeviation:F3}");
}

Console.WriteLine();
Console.WriteLine($"Накоплено ходов / оценено: {traces.Count()}");

// 5. Обучение на накопленном
IReadOnlyList<TrainingRound> sample = traces.ReadRated(elements);

Console.WriteLine($"Выборка: {sample.Count} ходов");
Console.WriteLine();

RouterTrainer routerTrainer = new(learningRate: 0.05f);
JudgeTrainer judgeTrainer = new(judge, learningRate: 0.5f);

double firstEpoch = 0, lastEpoch = 0;

for (int epoch = 0; epoch < 100; epoch++)
{
    double total = 0;

    foreach (TrainingRound round in sample)
    {
        total += routerTrainer.Train(round.Trace, round.Feedback);

        if (round.Requested is not null && round.Actual is not null)
            total += judgeTrainer.Train(round.Requested, round.Actual, round.Feedback.FeadbackScore);
    }

    if (epoch == 0) firstEpoch = total;
    lastEpoch = total;
}

Console.WriteLine($"Ошибка эпохи: {firstEpoch:F4} -> {lastEpoch:F4}");

// 6. Кого роутер выберет теперь на задачах разного типа
Console.WriteLine();
Console.WriteLine("Выбор после обучения:");

foreach ((string label, Style style, int symbols) in new[]
{
    ("детское объяснение", Style.Children, 600),
    ("научный обзор", Style.Scientific, 1500),
    ("деловое письмо", Style.OfficialBusiness, 700),
    ("техдокументация", Style.Technical, 1000)
})
{
    InputFeatures features = InputFeaturesService.GetFeatures(label);
    features.InputSpecifications = new Specifications { StyleType = style, SymbolLength = symbols, WordLength = symbols / 7 };

    string winner = Env.GetTopK(features, elements)[0].Element.Name!;
    Console.WriteLine($"  {label,-20} -> {winner}");
}

weights.Save(elements);
weights.Save(judge);

Console.WriteLine();
Console.WriteLine($"Веса и журнал ходов: {Path.GetFileName(dbPath)}");
