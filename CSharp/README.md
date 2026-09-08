<div align="center">

<img src="../docs/logo/logo.png" alt="FAIRouter" width="360">

### FAIRouter для .NET

Реализация на C#: библиотека `FAI.Router` и стенды для измерений

[![.NET](https://img.shields.io/badge/.NET-10.0-1E90FF?style=flat-square&logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![Stars](https://img.shields.io/github/stars/MASFractal/FAIRouter?style=flat-square&color=1E90FF&logo=github)](https://github.com/MASFractal/FAIRouter/stargazers)
[![License](https://img.shields.io/github/license/MASFractal/FAIRouter?style=flat-square&color=1E90FF)](../LICENSE)

</div>

Общее описание проекта и его назначение приведены в [корневом README](../README.md). Здесь собрано
только то, что касается сборки и устройства кода на C#.

## Раскладка

```
CSharp/
├── src/FAI.Router/          библиотека
│   ├── RoutedElements/      кандидат на исполнение: цены, скорость, обучаемый вектор
│   ├── JudgeLogic/          судья, спецификация, разбор расхождений
│   ├── Services/            извлечение ТЗ, замер факта, текстовые метрики
│   ├── LLM/                 обращения к модели и схемы ответа
│   ├── RotationTracking/    признаки хода, трассировка, отзыв
│   ├── Training/            обучение судьи и роутера
│   ├── Catalog/             цены и возможности моделей у поставщика
│   ├── Persistence/         веса и журнал ходов в SQLite
│   ├── Env.cs               среда соревнования кандидатов
│   └── Settings.cs          веса метрики, размерности, клиент модели
└── tests/MainTest/          демонстрация полного хода
```

Пространство имен совпадает с каталогом, отсюда `FAI.Router.JudgeLogic`, `FAI.Router.Training` и так
далее.

## Сборка

Понадобится **.NET 10** и репозиторий `AIFramework3Open`, который должен лежать рядом с `FAIRouter`.
Ссылки на него относительные, поэтому без соседнего репозитория проект не соберется.

```
GA/
├── FAIRouter/
└── AIFramework3Open/
```

```bash
dotnet build CSharp/src/FAI.Router/FAI.Router.csproj
```

| Ссылка | Зачем |
|---|---|
| `AI` | `Vector`, `Matrix` |
| `AI.LLM` | клиент OpenRouter, структурированный ответ по схеме |
| `AI.NLP` | разбиение на предложения с учетом русских сокращений |
| `AI.NeuralNetworks` | автоматическое дифференцирование и оптимизаторы (пространство имен `AI.ML.NeuralNetworks.V2`) |
| `Microsoft.Data.Sqlite` | хранилище весов и журнала ходов |

Единственным пакетом остается `Microsoft.Data.Sqlite`, остальное подключается ссылками на проекты.

## Ключевые типы

| Тип | Отвечает за |
|---|---|
| `FaiRouter` | фасад: весь контур одним объектом, от запроса до отзыва и обучения |
| `Env` | ход роутинга целиком: признаки, соревнование, трассировка |
| `BaseRoutedElement` | кандидат: цены, скорость, обучаемый `IdealMatchVector` |
| `Specifications` | спецификация, одна и та же для заказа и для факта |
| `Judge` | оценка близости факта к заказу и режим критика |
| `DiffSpec` | расхождения по каждому пункту ТЗ |
| `SpecInputService`, `SpecOutputService` | распознать ТЗ из запроса, измерить готовый ответ |
| `RouterTrainer`, `JudgeTrainer` | контрастивное обучение и повтор оценки человека |
| `QualityPrior` | начальные веса кандидата из заранее замеренного качества по типам задач |
| `ModelCatalog` | цены, окна и возможности моделей из каталога OpenRouter |
| `SqliteTraceStore`, `SqliteWeightsStore` | журнал ходов и веса в одном файле |

## Использование

Короткий путь через фасад:

```csharp
Settings.LLM = new LLMWithOpenRouterClient(new LLMOptions { ApiKey = "...", ModelName = "openai/gpt-4o-mini" });

FaiRouter router = new(candidates, (candidate, prompt) => AskModel(candidate.Name, prompt), "fai-router.db");
RouterAnswer answer = await router.AskAsync(prompt);

router.Feedback(answer.RoundId!.Value, score: 1.0);
router.Train(epochs: 10);
router.Save();
```

Сервер, совместимый с OpenAI, для подключения к OpenClaw есть в Python-версии; для C# он пока не
написан.

Тот же контур по частям:

```csharp
// Клиент модели: общий на процесс либо свой у каждого компонента
Settings.LLM = new LLMWithOpenRouterClient(new LLMOptions
{
    ApiKey = "...",
    ModelName = "openai/gpt-4o-mini"
});

// Кандидаты берутся из каталога: руками задается только скорость, ее поставщик не публикует
IReadOnlyList<ModelInfo> catalog = await ModelCatalog.FetchAsync();

BaseRoutedElement[] candidates =
[
    ModelCatalog.CreateElement(catalog.First(m => m.Id == "google/gemini-2.5-flash"), tokensPerSecond: 200),
    ModelCatalog.CreateElement(catalog.First(m => m.Id == "anthropic/claude-haiku-4.5"), tokensPerSecond: 60)
];

// Задача с картинкой: неспособные отсеются до сравнения оценок
Tracert visual = await Env.RouteAsync(prompt, candidates, required: Capability.Vision);

// Ход: признаки задачи, соревнование кандидатов, трассировка
Tracert trace = await Env.RouteAsync(prompt, candidates);

// Победитель ответил, измеряем результат и оцениваем
Specifications actual = await new SpecOutputService().GetSpecificationsAsync(answer);
double score = judge.Rate(trace, trace.RequestedSpec!, actual);

Console.WriteLine(Judge.Criticize(trace.RequestedSpec!, actual));

// Ход попадает в журнал, а отзыв дописывается позже, когда его дал человек
long id = traces.Append(trace, trace.RequestedSpec, actual, prompt);
traces.SetFeedback(id, new Feedback { FType = FeedbackType.Human, FeadbackScore = 1.0 });

// Обучение по накопленному
foreach (TrainingRound round in traces.ReadRated(candidates))
{
    routerTrainer.Train(round.Trace, round.Feedback);
    judgeTrainer.Train(round.Requested!, round.Actual!, round.Feedback.FeadbackScore);
}
```

Судить несколькими моделями одновременно тоже можно, поскольку клиент передается в компонент, а
`Settings.LLM` остается значением по умолчанию.

```csharp
StyleClassifier byGemini = new(geminiClient);
StyleClassifier byHaiku = new(haikuClient);
```

## Стенды и измерения

| Проект | Что делает |
|---|---|
| `tests/MainTest` | полный ход на одном запросе: роутинг, ТЗ, замер, критик |
| [`docs/research/harness/JudgeBenchmark`](../docs/research/harness/JudgeBenchmark) | сравнение моделей в роли судьи на размеченных текстах |
| [`docs/research/harness/LiveSession`](../docs/research/harness/LiveSession) | живой прогон: настоящие ходы, накопление, обучение |

Ключ OpenRouter берется из переменной `OPENROUTER_API_KEY` либо из файла `key.txt` рядом с
программой. В систему контроля версий ключ не попадает.

```bash
dotnet run --project docs/research/harness/LiveSession
```

Результаты измерений собраны в [docs/research](../docs/research).

## О чем стоит знать заранее

* **В AIFramework есть два разных типа `Tensor`**: алгебраический `AI.DataStructs.Algebraic.Tensor`
  и тензор автоматического дифференцирования `AI.ML.NeuralNetworks.V2.Tensor`. В файле
  `Training/TensorBridge.cs` для второго стоит псевдоним.
* **Каталог `AI.NeuralNetworks` и пространство имен `AI.ML.NeuralNetworks.V2` не совпадают.** Это не
  опечатка, а особенность соседнего репозитория.
* **Обучение идет в формате float32**, потому что тензоры считают только в нем. Обратная запись в
  `Vector` и `Matrix` проходит через `TensorBridge`.
* **Перегрузка `SendToLLM(string)` подставляет системным сообщением промпт клиента.** У общего
  `Settings.LLM` он не задан, из-за чего поставщик отвергает запрос с пустым содержимым. Поэтому
  внутри библиотеки сообщения собираются явно.
* **Размерность признаков вычисляется, а не задается константой.** Значение `Settings.FeaturesSpecDim`
  считается из перечисления `Style` и числа метрик. Если добавили поле в `Specifications`, добавьте
  и его масштаб, иначе оно попадет в вектор в сырых единицах.
