using AI.LLM.Services.LLM;
using FAI.Router;
using FAI.Router.JudgeLogic;
using FAI.Router.RotationTracking;
using FAI.Router.RoutedElements;
using FAI.Router.Services;

// Ключ берется из key.txt рядом с проектом (в git не попадает) или из переменной окружения
string keyFile = Path.Combine(AppContext.BaseDirectory, "key.txt");
string apiKey = File.Exists(keyFile)
    ? File.ReadAllText(keyFile).Trim()
    : Environment.GetEnvironmentVariable("OPENROUTER_API_KEY") ?? "";

Settings.LLM = new LLMWithOpenRouterClient(new LLMOptions
{
    ApiKey = apiKey,
    ModelName = "openai/gpt-4o-mini"
});

const string Prompt = "Напиши обзор методов кластеризации на 4000 знаков в научном стиле, "
    + "раздели на 4 раздела, добавь таблицу сравнения и ссылки на источники.";

// ---------- Ход роутинга ----------

BaseRoutedElement fast = new() { Name = "Gemini 2.5 Flash", TPS = 200, DPMTInp = 0.3, DPMTOutp = 2.5 };
BaseRoutedElement strong = new() { Name = "Sonnet 4.6", TPS = 60, DPMTInp = 3, DPMTOutp = 15 };
BaseRoutedElement heavy = new() { Name = "Opus 4.6", TPS = 20, DPMTInp = 15, DPMTOutp = 75 };

Tracert trace = await Env.RouteAsync(Prompt, [fast, strong, heavy]);

Console.WriteLine("========== ХОД РОУТИНГА ==========");
Console.WriteLine($"Победитель: {trace.Winner.Name}");
Console.WriteLine($"Порядок:    {string.Join(" > ", trace.TopKElements.Select(item => item.Name))}");
Console.WriteLine($"Признаки:   {trace.InputFeatureVector.Count} координат");

// ---------- Что модель поняла из запроса ----------

Specifications requested = await new SpecInputService().GetSpecificationsAsync(Prompt);

Console.WriteLine();
Console.WriteLine("========== ТЗ, РАСПОЗНАННОЕ ИЗ ЗАПРОСА ==========");
Console.WriteLine($"Стиль             {requested.StyleType}");
Console.WriteLine($"Символов          {requested.SymbolLength}");
Console.WriteLine($"Слов              {requested.WordLength}");
Console.WriteLine($"Разделов          {requested.SectionCount}");
Console.WriteLine($"Таблиц            {requested.TableCount}");
Console.WriteLine($"Абзацев           {requested.ParagraphCount}");
Console.WriteLine($"Читаемость        {requested.ReadabilityScore:F0}");
Console.WriteLine($"Терминология      {requested.TermDensity:F2}");
Console.WriteLine($"Формальность      {requested.FormalityScore:F2}");
Console.WriteLine($"Язык              {requested.Language}");
Console.WriteLine($"Ссылки            {requested.HasReferences}");

// ---------- Заведомо плохой ответ: судья должен это увидеть ----------

const string Answer = """
    # Кластеризация

    Короче, кластеризация - это когда мы делим данные на кучки. Есть разные способы.

    Самый простой - k-means. Берём k точек и двигаем их, пока не сойдётся. Работает быстро!

    Ещё есть иерархическая. Она строит дерево. Тоже ничего.
    """;

Specifications actual = await new SpecOutputService().GetSpecificationsAsync(Answer);

Console.WriteLine();
Console.WriteLine("========== ЗАМЕР ФАКТИЧЕСКОГО ОТВЕТА ==========");
Console.WriteLine($"Стиль             {actual.StyleType}");
Console.WriteLine($"Символов          {actual.SymbolLength}");
Console.WriteLine($"Слов              {actual.WordLength}");
Console.WriteLine($"Разделов          {actual.SectionCount}");
Console.WriteLine($"Таблиц            {actual.TableCount}");
Console.WriteLine($"Абзацев           {actual.ParagraphCount}");
Console.WriteLine($"Читаемость        {actual.ReadabilityScore:F0}");
Console.WriteLine($"Терминология      {actual.TermDensity:F2}");
Console.WriteLine($"Формальность      {actual.FormalityScore:F2}");
Console.WriteLine($"Язык              {actual.Language}");
Console.WriteLine($"Ссылки            {actual.HasReferences}");

// ---------- Вердикт судьи ----------

DiffSpec diff = Judge.Criticize(requested, actual);
double cosine = actual.FeaturesSpecificationVector.Cos(requested.FeaturesSpecificationVector);

Console.WriteLine();
Console.WriteLine("========== КРИТИК ==========");
Console.WriteLine(diff);
Console.WriteLine();
Console.WriteLine($"Среднее отклонение {diff.TotalDeviation:F3}");
Console.WriteLine($"Провалено пунктов  {diff.Mismatches.Count()} из {diff.Deviations.Count}");
Console.WriteLine($"Косинус с заказом  {cosine:F4}");
