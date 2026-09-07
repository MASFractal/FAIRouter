using System.Diagnostics;
using AI.LLM.Services.LLM;
using FAI.Router;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.LLM;
using FAI.Router.Services;

// Ключ: переменная окружения OPENROUTER_API_KEY либо файл key.txt рядом с программой
string keyFile = Path.Combine(AppContext.BaseDirectory, "key.txt");
string apiKey = Environment.GetEnvironmentVariable("OPENROUTER_API_KEY")
    ?? (File.Exists(keyFile) ? File.ReadAllText(keyFile).Trim() : "");

if (string.IsNullOrWhiteSpace(apiKey))
{
    Console.WriteLine("Нет ключа: задайте OPENROUTER_API_KEY или положите key.txt рядом с программой.");
    return;
}

string[] models =
[
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct"
];

// Тексты с заведомо известным стилем и заданным порядком по насыщенности терминами
(string Label, Style Style, int TermRank, string Text)[] texts =
[
    ("бытовой", Style.Conversational, 0,
        "Слушай, я вчера попробовал разбить данные на кучки. Ну прям как носки по цветам раскладываешь. Вроде получилось, доволен."),

    ("детский", Style.Children, 1,
        "Представь, что у тебя есть коробка с игрушками. Мишки лежат с мишками, машинки с машинками. Вот так же и компьютер раскладывает похожие вещи вместе. Здорово, правда?"),

    ("технический", Style.Technical, 2,
        "Алгоритм k-means минимизирует внутрикластерную сумму квадратов расстояний. На каждой итерации пересчитываются центроиды, пока разметка не стабилизируется. Сложность O(nkdi) на итерацию."),

    ("официально-деловой", Style.OfficialBusiness, 3,
        "Настоящим уведомляем, что в соответствии с пунктом 4.2 регламента обработка набора данных подлежит кластеризации силами подрядчика в срок до 15 числа отчётного месяца."),

    ("научный", Style.Scientific, 4,
        "Спектральная кластеризация опирается на собственные векторы нормализованного лапласиана графа сходства. Критерий Ng-Jordan-Weiss минимизирует нормализованный разрез, а спектральный зазор задаёт оценку числа компонент связности выборки.")
];

// Все шесть требований названы в запросе прямо: это проверка на верность, а не на догадливость
const string SpecPrompt = "Напиши обзор методов кластеризации на 4000 знаков в научном стиле, раздели на 4 раздела, добавь таблицу сравнения и ссылки на источники.";

Console.WriteLine($"Моделей {models.Length}, размеченных текстов {texts.Length}, плюс извлечение ТЗ");
Console.WriteLine();

List<(string Model, int StyleHits, double TermOrder, int SpecHits, int Failures, long Millis)> results = [];

foreach (string model in models)
{
    // Клиент один на процесс, поэтому модели сравниваются по очереди, а не параллельно
    Settings.LLM = new LLMWithOpenRouterClient(new LLMOptions { ApiKey = apiKey, ModelName = model });

    StyleClassifier classifier = new();
    SpecInputService specService = new();

    int styleHits = 0, specHits = 0, failures = 0;
    List<(int Rank, double Density)> densities = [];
    Stopwatch clock = Stopwatch.StartNew();

    Console.WriteLine($"--- {model} ---");

    foreach ((string label, Style expected, int rank, string text) in texts)
    {
        try
        {
            StyleAssessment assessment = await classifier.AssessAsync(text);
            bool hit = assessment.StyleType == expected;

            if (hit) styleHits++;
            densities.Add((rank, assessment.TermDensity));

            string verdict = hit ? "верно" : $"ждали {expected}";
            Console.WriteLine($"  {label,-20} {assessment.StyleType,-18} {verdict,-24} терм {assessment.TermDensity:F2}");
        }
        catch (Exception ex)
        {
            failures++;

            // Текст ошибки важнее факта отказа: 404 на слаге модели легко принять
            // за отсутствие поддержки структурированного ответа
            Console.WriteLine($"  {label,-20} ОТКАЗ: {Shorten(ex)}");
        }
    }

    try
    {
        Specifications spec = await specService.GetSpecificationsAsync(SpecPrompt);

        if (spec.SymbolLength == 4000) specHits++;
        if (spec.SectionCount == 4) specHits++;
        if (spec.TableCount == 1) specHits++;
        if (spec.HasReferences) specHits++;
        if (spec.StyleType == Style.Scientific) specHits++;
        if (spec.Language == "ru") specHits++;

        Console.WriteLine($"  ТЗ: знаков {spec.SymbolLength}, разделов {spec.SectionCount}, таблиц {spec.TableCount}, ссылки {spec.HasReferences}, стиль {spec.StyleType}, язык {spec.Language} -> {specHits}/6");
    }
    catch (Exception ex)
    {
        failures++;
        Console.WriteLine($"  ТЗ: ОТКАЗ {Shorten(ex)}");
    }

    clock.Stop();
    results.Add((model, styleHits, Correlation(densities), specHits, failures, clock.ElapsedMilliseconds));

    Console.WriteLine();
}

Console.WriteLine("================== ИТОГ ==================");
Console.WriteLine($"{"модель",-36} {"стиль",-8} {"термины",-9} {"ТЗ",-6} {"отказы",-8} время");

foreach ((string model, int styleHits, double order, int specHits, int failures, long millis) in results.OrderByDescending(item => item.StyleHits + item.SpecHits))
    Console.WriteLine($"{model,-36} {styleHits}/5      {order,6:F2}    {specHits}/6    {failures,-8} {millis / 1000.0:F1} с");

Console.WriteLine();
Console.WriteLine("стиль: совпадений с разметкой | термины: ранговая корреляция с ожидаемым порядком | ТЗ: явных требований извлечено точно");

// Начало сообщения провайдера, если оно есть: по нему видно причину отказа
static string Shorten(Exception exception)
{
    int start = exception.Message.IndexOf("\"message\"", StringComparison.Ordinal);

    if (start < 0)
        return exception.GetType().Name;

    return exception.Message[start..Math.Min(exception.Message.Length, start + 90)];
}

// Ранговая корреляция Спирмена между заданным и измеренным порядком
static double Correlation(List<(int Rank, double Density)> pairs)
{
    if (pairs.Count < 2) return 0;

    double[] expected = [.. pairs.Select(pair => (double)pair.Rank)];
    double[] measured = RankOf([.. pairs.Select(pair => pair.Density)]);

    double meanExpected = expected.Average(), meanMeasured = measured.Average();
    double covariance = 0, varianceExpected = 0, varianceMeasured = 0;

    for (int i = 0; i < expected.Length; i++)
    {
        covariance += (expected[i] - meanExpected) * (measured[i] - meanMeasured);
        varianceExpected += (expected[i] - meanExpected) * (expected[i] - meanExpected);
        varianceMeasured += (measured[i] - meanMeasured) * (measured[i] - meanMeasured);
    }

    return varianceExpected == 0 || varianceMeasured == 0
        ? 0
        : covariance / Math.Sqrt(varianceExpected * varianceMeasured);
}

static double[] RankOf(double[] values)
{
    double[] ranks = new double[values.Length];
    int[] order = [.. Enumerable.Range(0, values.Length).OrderBy(index => values[index])];

    for (int position = 0; position < order.Length; position++)
        ranks[order[position]] = position;

    return ranks;
}
