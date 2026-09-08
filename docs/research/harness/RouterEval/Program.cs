using System.Text.Json;
using AI.DataStructs.Algebraic;
using AI.LLM.Core.Models.Common.Messages;
using AI.LLM.Core.Models.Common.Responses;
using AI.LLM.Services.LLM;
using FAI.Router;
using FAI.Router.Catalog;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using FAI.Router.Services;
using FAI.Router.Training;

// Лучше ли роутер очевидных стратегий на задачах, которых он не видел.
// Обучение: замер по 8 задачам из model-quality-grid.md, из него строятся начальные веса.
// Проверка: 8 новых задач тех же типов. Каждая модель отвечает на каждую, и по сетке задним
// числом считается, что получила бы каждая стратегия.

string keyFile = Path.Combine(AppContext.BaseDirectory, "key.txt");
string apiKey = Environment.GetEnvironmentVariable("OPENROUTER_API_KEY")
    ?? (File.Exists(keyFile) ? File.ReadAllText(keyFile).Trim() : "");

if (string.IsNullOrWhiteSpace(apiKey))
{
    Console.WriteLine("Нет ключа: задайте OPENROUTER_API_KEY или положите key.txt рядом с программой.");
    return;
}

LLMBase Client(string model) => new LLMWithOpenRouterClient(new LLMOptions { ApiKey = apiKey, ModelName = model });
Settings.LLM = Client("openai/gpt-4o-mini");

// ---------- Кандидаты из каталога, скорость руками ----------

IReadOnlyList<ModelInfo> catalog = await ModelCatalog.FetchAsync();

(string Id, double Tps)[] ids =
[
    ("google/gemini-2.5-flash", 200),
    ("openai/gpt-4.1-mini", 90),
    ("anthropic/claude-haiku-4.5", 60)
];

BaseRoutedElement[] candidates = [.. ids.Select(item =>
    ModelCatalog.CreateElement(catalog.First(model => model.Id == item.Id), item.Tps))];
LLMBase[] clients = [.. ids.Select(item => Client(item.Id))];

// ---------- Обучающие данные: замер по типам задач ----------

(Style Style, int Symbols, double Terms, double Formality)[] trainKinds =
[
    (Style.Scientific,       1500, 0.70, 0.90),
    (Style.Children,          600, 0.05, 0.15),
    (Style.OfficialBusiness,  700, 0.35, 0.95),
    (Style.Technical,        1000, 0.45, 0.50)
];

// Строки в порядке кандидатов, столбцы в порядке типов
double[][] measured =
[
    [0.468, 0.726, 0.741, 0.572],
    [0.532, 0.675, 0.765, 0.607],
    [0.634, 0.634, 0.650, 0.723]
];

InputFeatures TaskOf(Style style, int symbols, double terms, double formality)
{
    InputFeatures features = InputFeaturesService.GetFeatures("задача");
    features.InputSpecifications = new Specifications
    {
        StyleType = style, SymbolLength = symbols, WordLength = symbols / 7, SectionCount = 3,
        TermDensity = terms, FormalityScore = formality
    };
    return features;
}

Vector[] trainVectors = [.. trainKinds.Select(k => TaskOf(k.Style, k.Symbols, k.Terms, k.Formality).FeatureVector)];
Settings.TaskMean = trainVectors.Aggregate((a, b) => a + b) / trainVectors.Length;

for (int i = 0; i < candidates.Length; i++)
{
    candidates[i].IdealMatchVector = QualityPrior.FromMeasurements(
        [.. Enumerable.Range(0, trainKinds.Length).Select(k => (trainVectors[k], measured[i][k]))]);

    // Замер и есть опыт: 8 оцененных задач на кандидата, разброс оценок из той же таблицы
    double[] row = measured[i];
    double mean = row.Average();
    candidates[i].Experience = 8;
    candidates[i].ScoreVariance = row.Sum(v => (v - mean) * (v - mean)) / (row.Length - 1);
}

// ---------- Проверочные задачи: новые темы тех же типов ----------

(string Kind, string Prompt)[] tests =
[
    ("научный", "Напиши научный обзор методов регуляризации нейросетей на 1500 знаков, раздели на 3 раздела, добавь таблицу сравнения и ссылки на источники."),
    ("научный", "Подготовь научное описание градиентного бустинга на 1500 знаков, 3 раздела, с формулами и ссылками на источники."),
    ("детский", "Объясни восьмилетнему ребенку простыми словами, что такое интернет. Уложись в 600 знаков, без терминов."),
    ("детский", "Расскажи ребенку семи лет, что такое робот. 600 знаков, простыми словами и с примером из жизни."),
    ("деловой", "Составь официальное уведомление контрагенту о смене банковских реквизитов. 700 знаков, деловой стиль."),
    ("деловой", "Напиши официальное письмо заказчику о продлении срока действия договора. 700 знаков, деловой стиль."),
    ("технический", "Опиши алгоритм Дейкстры для технической документации: 1000 знаков, с блоком кода на Python и списком параметров."),
    ("технический", "Опиши устройство очереди сообщений для технической документации: 1000 знаков, с блоком кода и списком параметров.")
];

SpecInputService extractor = new();
SpecOutputService measurer = new();

// Ответы моделей и их замеры кешируются: решение роутера пересчитывается бесплатно,
// а обращения к моделям делаются один раз
string cachePath = Path.Combine(AppContext.BaseDirectory, "grid-cache.json");
GridRow[]? cached = File.Exists(cachePath)
    ? JsonSerializer.Deserialize<GridRow[]>(File.ReadAllText(cachePath))
    : null;

bool useCache = cached is not null && cached.Length == tests.Length
    && cached.All(row => row.Candidates.SequenceEqual(ids.Select(item => item.Id)));

// quality[задача][кандидат], cost[задача][кандидат]
double[][] quality = new double[tests.Length][];
double[][] cost = new double[tests.Length][];
int[] routerGreedy = new int[tests.Length];
double[][] routerProbability = new double[tests.Length][];
Specifications[] requestedSpecs = new Specifications[tests.Length];
List<GridRow> rows = [];

Console.WriteLine($"Кандидатов {candidates.Length}, проверочных задач {tests.Length}, " +
    (useCache ? "ответы из кеша" : $"ответов {candidates.Length * tests.Length}"));
Console.WriteLine();

for (int t = 0; t < tests.Length; t++)
{
    (string kind, string prompt) = tests[t];

    Specifications requested = useCache
        ? cached![t].Requested
        : await extractor.GetSpecificationsAsync(prompt);
    requestedSpecs[t] = requested;

    // Что решил бы роутер, глядя только на распознанное задание
    InputFeatures features = InputFeaturesService.GetFeatures(prompt);
    features.InputSpecifications = requested;

    List<(double Score, BaseRoutedElement Element)> ranked = Env.GetTopK(features, candidates);
    routerGreedy[t] = Array.IndexOf(candidates, ranked[0].Element);
    routerProbability[t] = Probabilities(ranked, candidates);

    if (useCache)
    {
        quality[t] = cached![t].Quality;
        cost[t] = cached[t].Cost;
    }
    else
    {
        quality[t] = new double[candidates.Length];
        cost[t] = new double[candidates.Length];

        for (int c = 0; c < candidates.Length; c++)
        {
            ChatCompletionsResponse response = await clients[c].SendToLLMFull([new LLMMessage(LLMMessage.UserRole, prompt)]);
            string answer = response.Choices[0].Message.Content?.ToString() ?? "";

            Specifications actual = await measurer.GetSpecificationsAsync(answer);
            quality[t][c] = 1 - Judge.Criticize(requested, actual).TotalDeviation;
            cost[t][c] = (candidates[c].DPMTInp * response.Usage.PromptTokens
                        + candidates[c].DPMTOutp * response.Usage.CompletionTokens) * 1e-6;
        }

        rows.Add(new GridRow(kind, prompt, [.. ids.Select(item => item.Id)], requested, quality[t], cost[t]));
    }

    int oracle = Array.IndexOf(quality[t], quality[t].Max());
    Console.WriteLine($"{kind,-12} роутер {Short(routerGreedy[t]),-8} лучший {Short(oracle),-8} " +
        $"качество {string.Join(" ", quality[t].Select(q => q.ToString("F2")))}   " +
        $"цена $ {string.Join(" ", cost[t].Select(v => v.ToString("F5")))}");
}

if (!useCache)
    File.WriteAllText(cachePath, JsonSerializer.Serialize(rows));

// ---------- Стратегии, оцененные по сетке ----------

int cheapest = Array.IndexOf(candidates, candidates.MinBy(c => c.DPMTOutp)!);
int strongest = Array.IndexOf(candidates, candidates.MaxBy(c => c.DPMTOutp)!);

(string Name, Func<int, double[]> Weights)[] strategies =
[
    ("всегда самая дешевая", t => OneHot(cheapest)),
    ("всегда самая дорогая", t => OneHot(strongest)),
    ("случайный выбор", t => Uniform()),
    ("роутер, жадный выбор", t => OneHot(routerGreedy[t])),
    ("роутер, сэмплирование", t => routerProbability[t]),
    ("идеальный выбор", t => OneHot(Array.IndexOf(quality[t], quality[t].Max())))
];

Console.WriteLine();
Console.WriteLine("=== Итог по 8 проверочным задачам ===");
Console.WriteLine($"{"стратегия",-26} {"качество",-10} {"цена $",-10} качество на цент");

foreach ((string name, Func<int, double[]> weights) in strategies)
{
    double meanQuality = 0, meanCost = 0;

    for (int t = 0; t < tests.Length; t++)
    {
        double[] w = weights(t);
        meanQuality += Enumerable.Range(0, candidates.Length).Sum(c => w[c] * quality[t][c]) / tests.Length;
        meanCost += Enumerable.Range(0, candidates.Length).Sum(c => w[c] * cost[t][c]) / tests.Length;
    }

    Console.WriteLine($"{name,-26} {meanQuality,-10:F3} {meanCost,-10:F5} {meanQuality / (meanCost * 100),8:F2}");
}

// Те же веса метрики, что профили ECO, AUTO и PREMIUM у ClawRouter: доли важности качества,
// цены и времени. Решения пересчитываются по кешу, обращений к моделям нет.
Console.WriteLine();
Console.WriteLine("=== Роутер при разных долях важности, жадный выбор ===");
Console.WriteLine($"{"профиль",-24} {"WQ / WC / Wt",-16} {"качество",-10} {"цена $",-10} качество на цент");

foreach ((string profile, double wq, double wc, double wt) in new[]
{
    ("качество прежде всего", 0.8, 0.1, 0.1),
    ("по умолчанию", 0.5, 0.25, 0.25),
    ("экономный", 0.3, 0.6, 0.1),
    ("только цена", 0.0, 1.0, 0.0)
})
{
    Settings.WQ = wq;
    Settings.WC = wc;
    Settings.Wt = wt;

    double profileQuality = 0, profileCost = 0;

    for (int t = 0; t < tests.Length; t++)
    {
        InputFeatures f = InputFeaturesService.GetFeatures(tests[t].Prompt);
        f.InputSpecifications = requestedSpecs[t];
        int pick = Array.IndexOf(candidates, Env.GetTopK(f, candidates)[0].Element);

        profileQuality += quality[t][pick] / tests.Length;
        profileCost += cost[t][pick] / tests.Length;
    }

    Console.WriteLine($"{profile,-24} {$"{wq:F1} / {wc:F2} / {wt:F2}",-16} {profileQuality,-10:F3} {profileCost,-10:F5} {profileQuality / (profileCost * 100),8:F2}");
}

Settings.WQ = 0.5;
Settings.WC = 0.25;
Settings.Wt = 0.25;

int hits = Enumerable.Range(0, tests.Length).Count(t => routerGreedy[t] == Array.IndexOf(quality[t], quality[t].Max()));
Console.WriteLine();
Console.WriteLine($"Роутер угадал лучшего в {hits} задачах из {tests.Length}");

double[] OneHot(int index) => [.. Enumerable.Range(0, candidates.Length).Select(c => c == index ? 1.0 : 0.0)];
double[] Uniform() => [.. Enumerable.Repeat(1.0 / candidates.Length, candidates.Length)];
string Short(int index) => candidates[index].Name!.Split('/')[1].Split('-')[0];

// Вероятности сэмплирования: тот же softmax с температурой, что в Env.Choose
static double[] Probabilities(List<(double Score, BaseRoutedElement Element)> ranked, BaseRoutedElement[] order)
{
    double temperature = Env.Temperature(ranked.Select(item => item.Element));
    double top = ranked[0].Score;
    double[] weights = [.. ranked.Select(item => temperature < 1e-9 ? (item.Score == top ? 1 : 0) : Math.Exp((item.Score - top) / temperature))];
    double total = weights.Sum();

    double[] result = new double[order.Length];
    for (int i = 0; i < ranked.Count; i++)
        result[Array.IndexOf(order, ranked[i].Element)] = weights[i] / total;

    return result;
}

// Строка сетки: задача, распознанное задание и что дала каждая модель
record GridRow(string Kind, string Prompt, string[] Candidates, Specifications Requested, double[] Quality, double[] Cost);
