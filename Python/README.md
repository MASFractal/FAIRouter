# FAIRouter на Python

Перенос библиотеки `FAI.Router` с C# на Python. Устройство, метрика и все находки измерений
общие, они описаны в [корневом README](../README.md) и в [docs/research](../docs/research).
Здесь только то, что касается этой версии.

## Установка

Нужен Python 3.11 и выше. Зависимость одна: `numpy` для векторов и матриц. Обращения к
OpenRouter идут через встроенный `urllib`, база на встроенном `sqlite3`.

```bash
pip install -e ".[test]"
pytest
```

## Отличия от версии на C#

* **Градиенты выведены аналитически на numpy**, без автограда. В C# обучение считал
  `AI.NeuralNetworks`, потому что лежал рядом. Здесь оба градиента умещаются в несколько строк:
  у судьи это производная косинуса по матрице, у роутера производная порогового штрафа по
  векторам. Правила обучения совпадают с C#-версией один в один.
* **Разбиение на предложения** делает свой короткий список русских сокращений вместо
  `SentencesTokenizer` из `AI.NLP`. На типовых текстах результат тот же, на редких сокращениях
  может расходиться.
* **Вызовы синхронные.** Обращения к модели идут через `urllib`, без `asyncio`.
* Имена в стиле Python: `RoutedElement` вместо `BaseRoutedElement`, `env.route` вместо
  `Env.RouteAsync`, `env.choose`, `env.get_top_k`, `env.execute`.

## Раскладка

```
fai_router/
├── enums.py            Style, FeedbackType, Capability
├── settings.py         веса метрики, температура, среднее по задачам, клиент модели
├── specifications.py   спецификация с масштабами и вектором признаков
├── diff_spec.py        разбор расхождений по пунктам (критик)
├── judge.py            судья: оценка, проставление в трассировку, критик
├── routed_element.py   кандидат: цены, скорость, возможности, обучаемый вектор
├── tracking.py         InputFeatures, Tracert, Feedback
├── text_metrics.py     замер структуры текста без модели
├── services.py         извлечение задания и замер ответа
├── env.py              ход роутинга, сэмплирование с температурой, отсев, запасной вариант
├── llm/                клиент OpenRouter, распознавание задания и стиля
├── training/           обучение судьи и роутера, начальные веса из замера
├── persistence/        веса и журнал ходов в SQLite
└── catalog.py          цены и возможности моделей у поставщика
```

## Использование

```python
from fai_router import Settings, Judge, env
from fai_router.llm import OpenRouterClient
from fai_router import catalog
from fai_router.services import SpecOutputService
from fai_router.persistence import SqliteTraceStore

Settings.llm = OpenRouterClient(api_key="...", model="openai/gpt-4o-mini")

models = catalog.fetch()
candidates = [
    catalog.create_element(next(m for m in models if m.id == "google/gemini-2.5-flash"), tokens_per_second=200),
    catalog.create_element(next(m for m in models if m.id == "anthropic/claude-haiku-4.5"), tokens_per_second=60),
]

trace = env.route(prompt, candidates)
answer = env.execute(trace, lambda candidate: ask_model(candidate.name, prompt))

actual = SpecOutputService().get_specifications(answer)
judge = Judge()
judge.rate(trace, trace.requested_spec, actual)
print(Judge.criticize(trace.requested_spec, actual))

traces = SqliteTraceStore("weights.db")
round_id = traces.append(trace, trace.requested_spec, actual, prompt)
```

Тесты повторяют проверки, сделанные для версии на C#, с теми же числами: косинус 0,6408 между
научным и детским текстом после нормировки, 9 провалов из 16 у критика, четыре совпадения из
четырех у начальных весов из замера, разделение трех типов задач на 20 запусках.

## Живой прогон

Стенд [docs/research/harness/LiveSessionPython](../docs/research/harness/LiveSessionPython)
зеркалит стенд на C#: шесть настоящих задач, три модели, полный контур от распознавания задания
до обучения на журнале. Ключ OpenRouter берется из переменной `OPENROUTER_API_KEY` либо из файла
`key.txt` рядом со стендом.

```bash
python docs/research/harness/LiveSessionPython/live_session.py
```

Результат и сравнение с C#-версией записаны в
[docs/research/live-session.md](../docs/research/live-session.md): стиль распознан в шести ходах
из шести, ходы разошлись по трем кандидатам, самозакрепления нет.
