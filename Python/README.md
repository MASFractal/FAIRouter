# FAIRouter на Python

Перенос библиотеки `FAI.Router` с C# на Python. Устройство, метрика и все находки измерений
общие, они описаны в [корневом README](https://github.com/MASFractal/FAIRouter/blob/main/README.md) и в [docs/research](https://github.com/MASFractal/FAIRouter/tree/main/docs/research).
Здесь только то, что касается этой версии.

## Установка

Нужен Python 3.10 и выше. Зависимость одна: `numpy` для векторов и матриц. Обращения к
поставщику моделей идут через встроенный `urllib`, база на встроенном `sqlite3`.

```bash
pip install fai-router
```

Пока пакет не выложен на PyPI, та же команда ставит его из репозитория:

```bash
pip install "git+https://github.com/MASFractal/FAIRouter#subdirectory=Python"
```

После установки файл с вызовом роутера может лежать где угодно. Без установки его кладут в этот
каталог, `Python`, рядом с пакетом `fai_router`, иначе `import fai_router` пакет не найдет.

Для разработки пакет ставится в режиме правки вместе с тестами, сборка колеса и выкладка делаются
обычными средствами:

```bash
pip install -e ".[test]"
pytest
python -m build            # dist/fai_router-*.whl и .tar.gz
python -m twine upload dist/*
```

Выкладку на PyPI делает и действие `.github/workflows/publish-python.yml` по тегу вида `py-v0.1.0`;
для него в настройках проекта на pypi.org один раз включается доверенная публикация из этого
репозитория. Снимок замеров `fai_router/data/benchmark-snapshot.json` входит в пакет, а забор
внешних замеров `fai_router/sources` в него не входит.

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
  `Env.RouteAsync`, `env.choose`, `env.get_top_k`, `env.get_sufficient`, `env.execute`.

## Раскладка

```
fai_router/
├── enums.py            Style, Domain, ProgrammingLanguage, ScienceField, TaskKind, FeedbackType, Capability
├── settings.py         веса метрики, планка, температура, среднее по задачам, клиент модели
├── specifications.py   спецификация с масштабами и вектором признаков
├── diff_spec.py        разбор расхождений по пунктам (критик)
├── judge.py            судья: оценка, проставление в трассировку, критик
├── routed_element.py   кандидат: цены, скорость, возможности, обучаемый вектор
├── tracking.py         InputFeatures, Tracert, Feedback
├── text_metrics.py     замер структуры текста без модели
├── services.py         извлечение задания и замер ответа
├── env.py              ход роутинга, сэмплирование, выбор с планкой, отсев, запасной вариант
├── llm/                клиент поставщика (FractalRouter, OpenRouter), распознавание задания и стиля, судья содержания
├── training/           обучение судьи и роутера, калибровка, начальные веса из замера и из снимка
├── persistence/        веса и журнал ходов в SQLite
├── catalog.py          цены и возможности моделей из каталога FractalRouter или OpenRouter
├── content_review.py   оценка содержания: критерии, проверяемые утверждения, замечания
├── benchmarks.py       снимок внешних замеров и сопоставление имен
└── data/               профили задач по сериям замеров, общие с версией на C#
```

## Фасад и сервер

`FaiRouter` собирает весь контур в один объект: выбор, выполнение с запасным вариантом, замер,
оценка судьи, журнал, отзывы и обучение.

```python
from fai_router import FaiRouter

router = FaiRouter.from_fractalrouter("rtr_live_...", ["google/gemini-2.5-flash", "openai/gpt-4.1-mini"],
                                      database_path="fai-router.db")
answer = router.ask("Напиши научный обзор на 1500 знаков")
router.feedback(answer.round_id, 1.0)   # отзыв от 0 (плохо) до 1 (отлично); 0.2 означает плохой ответ
router.train(epochs=10)
router.save()
```

Модели можно не называть: тогда выбор идет среди популярных моделей из комплекта
(`fai_router/data/popular_models.json`, 54 модели с рейтингами и недорогие рабочие лошадки). Вместо
списка принимаются наборы `"popular"` и `"all"` (весь каталог поставщика). Профиль весов `weights`
(`"quality"`, `"balance"`, `"price"` или свои `RouteWeights`) задается роутеру на все ходы либо
отдельному ходу в `ask`. Таблица моделей и описание профилей есть в корневом README.

Три фабрики: `from_fractalrouter` для [FractalRouter](https://fractalrouter.ru) (наш шлюз к
моделям с оплатой в рублях, не путать с FractalGPT), `from_openrouter` для OpenRouter и
`from_openai_compatible(base_url, ...)` для любого сервера по протоколу OpenAI chat completions.
Через одного поставщика и один ключ идут и ответы кандидатов, и работа судьи (`judge_model`). Цены
берутся из каталога поставщика по идентификатору модели, у FractalRouter в рублях, у OpenRouter в
долларах; для выбора важны только отношения цен кандидатов. Модели, которой в каталоге нет, цену
задает довод `prices={"имя": (вход, выход)}` за миллион токенов. Начальные веса кандидатов
приходят из снимка замеров в комплекте пакета (`fai_router/data/benchmark-snapshot.json`), пустая
база на старте в порядке.

Клиент `OpenRouterClient` принимает `base_url` и повторяет запрос при обрыве соединения, таймауте
и ответах 429 и 5xx; после исчерпания повторов поднимает `LlmRequestError` с именем поставщика и
модели. Сбой судьи ход не роняет: ответ исполнителя возвращается без оценки.

Исполнителя можно задать своего: функция получает кандидата и сообщения диалога и возвращает
текст либо `Completion` с расходом токенов.

`fai_router.server` поднимает сервер, совместимый с OpenAI chat completions, для OpenClaw и любого
другого клиента:

```bash
python -m fai_router.server --models google/gemini-2.5-flash,openai/gpt-4.1-mini --db fai-router.db
```

Подключение к OpenClaw описано в [корневом README](https://github.com/MASFractal/FAIRouter/blob/main/README.md).

## Использование

```python
from fai_router import Settings, Judge, env
from fai_router.llm import OpenRouterClient
from fai_router import benchmarks, catalog
from fai_router.services import SpecOutputService
from fai_router.persistence import SqliteTraceStore

Settings.llm = OpenRouterClient(api_key="...", model="openai/gpt-4o-mini")

models = catalog.fetch()
# Снимок замеров: кандидат стартует с прогноза по сериям снимка
snapshot = benchmarks.default_snapshot()   # снимок из комплекта пакета, fai_router/data/benchmark-snapshot.json
candidates = [
    catalog.create_element(next(m for m in models if m.id == "google/gemini-2.5-flash"), tokens_per_second=200, benchmarks=snapshot),
    catalog.create_element(next(m for m in models if m.id == "anthropic/claude-haiku-4.5"), tokens_per_second=60, benchmarks=snapshot),
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
научным и детским текстом после нормировки, 9 провалов из 20 у критика, четыре совпадения из
четырех у начальных весов из замера, разделение трех типов задач на 20 запусках. Калибровка
планки совпадает с версией на C# до девятого знака.

## Живой прогон

Стенд [docs/research/harness/LiveSessionPython](https://github.com/MASFractal/FAIRouter/tree/main/docs/research/harness/LiveSessionPython)
зеркалит стенд на C#: шесть настоящих задач, три модели, полный контур от распознавания задания
до обучения на журнале. Стенд ходит в FractalRouter по ключу из `FRACTALROUTER_API_KEY` либо в
OpenRouter по ключу из `OPENROUTER_API_KEY`; без переменных ключ берется из файла `key.txt` рядом
со стендом, и поставщик опознается по виду ключа.

```bash
python docs/research/harness/LiveSessionPython/live_session.py
```

Результат и сравнение с C#-версией записаны в
[docs/research/live-session.md](https://github.com/MASFractal/FAIRouter/blob/main/docs/research/live-session.md): стиль распознан в шести ходах
из шести, ходы разошлись по трем кандидатам, самозакрепления нет.
