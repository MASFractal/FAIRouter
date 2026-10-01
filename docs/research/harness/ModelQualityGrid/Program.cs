using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Services.LLM;
using FAI.Router;
using FAI.Router.JudgeLogic;
using FAI.Router.Services;

// Различается ли качество моделей по типам задач. Роутинга здесь нет: каждая модель отвечает
// на каждую задачу, и качество каждого ответа измеряется судьей. Если разброс внутри типа
// близок к нулю, роутеру нечему учиться и обучение бессмысленно.

string keyFile = Path.Combine(AppContext.BaseDirectory, "key.txt");
string apiKey = Environment.GetEnvironmentVariable("OPENROUTER_API_KEY")
    ?? (File.Exists(keyFile) ? File.ReadAllText(keyFile).Trim() : "");

if (string.IsNullOrWhiteSpace(apiKey))
{
    Console.WriteLine("Нет ключа: задайте OPENROUTER_API_KEY или положите key.txt рядом с программой.");
    return;
}

LLMBase Client(string model) => new LLMWithOpenRouterClient(new LLMOptions { ApiKey = apiKey, ModelName = model });

// Распознает задачу и разбирает ответы одна модель, чтобы измерение было единообразным
Settings.LLM = Client("openai/gpt-4o-mini");

(string Name, LLMBase Llm)[] models =
[
    ("gemini-2.5-flash", Client("google/gemini-2.5-flash")),
    ("gpt-4.1-mini",     Client("openai/gpt-4.1-mini")),
    ("claude-haiku-4.5", Client("anthropic/claude-haiku-4.5"))
];

(string Kind, string Prompt)[] tasks =
[
    ("научный",  "Напиши научный обзор методов кластеризации на 1500 знаков, раздели на 3 раздела, добавь таблицу сравнения и ссылки на источники."),
    ("научный",  "Подготовь научное описание метода главных компонент на 1500 знаков, 3 раздела, с формулами и ссылками на источники."),
    ("детский",  "Объясни восьмилетнему ребенку простыми словами, что такое кластеризация. Уложись в 600 знаков, без терминов."),
    ("детский",  "Расскажи ребенку семи лет, что такое нейросеть. 600 знаков, простыми словами и с примером из жизни."),
    ("деловой",  "Составь официальное уведомление подрядчику о переносе сроков сдачи отчета. 700 знаков, деловой стиль."),
    ("деловой",  "Напиши официальное письмо заказчику о результатах приемки работ. 700 знаков, деловой стиль."),
    ("технический", "Опиши алгоритм k-means для технической документации: 1000 знаков, с блоком кода на Python и списком параметров."),
    ("технический", "Опиши устройство HTTP-кеша для технической документации: 1000 знаков, с блоком кода и списком заголовков.")
];

SpecInputService extractor = new();
SpecOutputService measurer = new();

Console.WriteLine($"Моделей {models.Length}, задач {tasks.Length}, всего ответов {models.Length * tasks.Length}");
Console.WriteLine();

Dictionary<(string Kind, string Model), List<double>> quality = [];

foreach ((string kind, string prompt) in tasks)
{
    // ТЗ распознается один раз на задачу и общее для всех моделей: иначе разброс распознавания
    // подмешался бы в сравнение самих моделей
    Specifications requested = await extractor.GetSpecificationsAsync(prompt);

    foreach ((string name, LLMBase llm) in models)
    {
        string answer = await llm.SendToLLM([new LLMMessage(LLMMessage.UserRole, prompt)]);
        Specifications actual = await measurer.GetSpecificationsAsync(answer);
        DiffSpec diff = Judge.Criticize(requested, actual);
        double score = 1 - diff.TotalDeviation;

        if (!quality.TryGetValue((kind, name), out List<double>? scores))
            quality[(kind, name)] = scores = [];

        scores.Add(score);

        Console.WriteLine($"{kind,-12} {name,-18} качество {score:F3}   знаков {actual.SymbolLength,5} из {requested.SymbolLength,5}   провалено {diff.Mismatches.Count()} из {diff.Deviations.Count}");
    }
}

string[] kinds = [.. tasks.Select(task => task.Kind).Distinct()];

Console.WriteLine();
Console.WriteLine("=== Качество по типам задач (среднее по двум задачам) ===");
Console.WriteLine($"{"тип задачи",-14} {string.Join("  ", models.Select(model => $"{model.Name,-18}"))} разброс  лучший");

foreach (string kind in kinds)
{
    double[] scores = [.. models.Select(model => quality[(kind, model.Name)].Average())];
    double spread = scores.Max() - scores.Min();
    string best = models[Array.IndexOf(scores, scores.Max())].Name;

    Console.WriteLine($"{kind,-14} {string.Join("  ", scores.Select(s => $"{s,-18:F3}"))} {spread:F3}    {best}");
}

Console.WriteLine();
Console.WriteLine("=== Есть ли чему учиться ===");

double[] byModel = [.. models.Select(model => kinds.Average(kind => quality[(kind, model.Name)].Average()))];
double overallSpread = byModel.Max() - byModel.Min();
double meanSpreadInKind = kinds.Average(kind =>
{
    double[] scores = [.. models.Select(model => quality[(kind, model.Name)].Average())];
    return scores.Max() - scores.Min();
});
int distinctWinners = kinds.Select(kind =>
    models[Array.IndexOf(
        models.Select(m => quality[(kind, m.Name)].Average()).ToArray(),
        models.Max(m => quality[(kind, m.Name)].Average()))].Name).Distinct().Count();

Console.WriteLine($"Разброс между моделями в среднем по всем типам: {overallSpread:F3}");
Console.WriteLine($"Средний разброс внутри одного типа задач:       {meanSpreadInKind:F3}");
Console.WriteLine($"Разных победителей по типам задач:              {distinctWinners} из {kinds.Length}");
Console.WriteLine();
Console.WriteLine(distinctWinners > 1
    ? "У разных типов задач разные лучшие модели: роутеру есть чему учиться."
    : "Победитель один на всех типах: выбор по типу задачи ничего не дает, достаточно выбирать по цене.");
