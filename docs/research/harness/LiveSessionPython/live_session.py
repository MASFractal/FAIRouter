"""Живой прогон на Python: настоящие ходы настоящими моделями, оценка судьи, накопление
и обучение на накопленном. Зеркало стенда LiveSession на C#: те же задачи, модели и цены."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "Python"))

from fai_router import Judge, Settings, env  # noqa: E402
from fai_router.enums import FeedbackType, Style  # noqa: E402
from fai_router.llm import OpenRouterClient  # noqa: E402
from fai_router.persistence import SqliteTraceStore, SqliteWeightsStore  # noqa: E402
from fai_router.routed_element import RoutedElement  # noqa: E402
from fai_router.services import InputFeaturesService, SpecOutputService  # noqa: E402
from fai_router.specifications import Specifications  # noqa: E402
from fai_router.tracking import Feedback  # noqa: E402
from fai_router.training import JudgeTrainer, RouterTrainer  # noqa: E402

HERE = Path(__file__).resolve().parent
key_file = HERE / "key.txt"
api_key = os.environ.get("OPENROUTER_API_KEY") or (key_file.read_text().strip() if key_file.exists() else "")
if not api_key:
    print("Нет ключа: задайте OPENROUTER_API_KEY или положите key.txt рядом с программой.")
    sys.exit(1)


def client(model: str) -> OpenRouterClient:
    return OpenRouterClient(api_key, model)


# Распознаванием задачи и разбором ответа занимается одна модель, по измерению она лучшая
Settings.llm = client("openai/gpt-4o-mini")

# Кандидаты: цены и скорость те же, что в стенде на C#
candidates = [
    (RoutedElement("gemini-2.5-flash", tps=200, dpmt_inp=0.30, dpmt_outp=2.50), client("google/gemini-2.5-flash")),
    (RoutedElement("gpt-4.1-mini", tps=90, dpmt_inp=0.40, dpmt_outp=1.60), client("openai/gpt-4.1-mini")),
    (RoutedElement("claude-haiku-4.5", tps=60, dpmt_inp=1.00, dpmt_outp=5.00), client("anthropic/claude-haiku-4.5")),
]
elements = [element for element, _ in candidates]
client_of = {element.name: llm for element, llm in candidates}

prompts = [
    "Объясни восьмилетнему ребёнку простыми словами, что такое кластеризация. Уложись в 600 знаков.",
    "Напиши краткий научный обзор методов кластеризации на 1500 знаков, раздели на 3 раздела, добавь таблицу сравнения.",
    "Составь официальное уведомление подрядчику о сроках сдачи отчёта. 700 знаков, деловой стиль.",
    "Опиши алгоритм k-means для технической документации: 1000 знаков, с блоком кода на Python.",
    "Расскажи в разговорном стиле, зачем нужна кластеризация. 500 знаков, без формул.",
    "Подготовь научное описание спектральной кластеризации на 1200 знаков со ссылками на источники.",
]

db_path = HERE / "live-session.db"
if db_path.exists():
    db_path.unlink()

traces = SqliteTraceStore(str(db_path))
weights = SqliteWeightsStore(str(db_path))
measurer = SpecOutputService()
judge = Judge()

print(f"Кандидатов {len(elements)}, запросов {len(prompts)}")
print()

for prompt in prompts:
    # 1. Ход роутинга: признаки задачи и выбор исполнителя
    trace = env.route(prompt, elements)

    # 2. Победитель действительно отвечает, при отказе ход переходит следующему
    answer = env.execute(trace, lambda candidate: client_of[candidate.name].complete(
        [{"role": "user", "content": prompt}]))

    # 3. Замер факта и оценка судьи, оценка идет в саму трассировку
    requested = trace.requested_spec
    actual = measurer.get_specifications(answer)
    score = judge.rate(trace, requested, actual)
    diff = Judge.criticize(requested, actual)

    # 4. Ход в журнал, отзыв ставится автоматически по разбору критика
    round_id = traces.append(trace, requested, actual, prompt)
    traces.set_feedback(round_id, Feedback(FeedbackType.AUTO, 1 - diff.total_deviation))

    print(f"[{round_id}] {trace.winner.name:<18} {requested.style_type.value:<18} -> {actual.style_type.value:<18} "
          f"знаков {actual.symbol_length:>5} из {requested.symbol_length:>5}   судья {score:.3f}   "
          f"отзыв {1 - diff.total_deviation:.3f}{'   разведка' if trace.is_exploration else ''}")

print()
print(f"Накоплено ходов / оценено: {traces.count()}")

# 5. Обучение на накопленном
sample = traces.read_rated(elements)
print(f"Выборка: {len(sample)} ходов")
print()

router_trainer = RouterTrainer(learning_rate=0.05)
judge_trainer = JudgeTrainer(judge, learning_rate=0.5)
first_epoch = last_epoch = 0.0

for epoch in range(100):
    total = 0.0
    for training_round in sample:
        total += router_trainer.train(training_round.trace, training_round.feedback)
        if training_round.requested is not None and training_round.actual is not None:
            total += judge_trainer.train(training_round.requested, training_round.actual,
                                         training_round.feedback.score)
    if epoch == 0:
        first_epoch = total
    last_epoch = total

print(f"Ошибка эпохи: {first_epoch:.4f} -> {last_epoch:.4f}")

# 6. Кого роутер выберет теперь на задачах разного типа
print()
print("Выбор после обучения:")
for label, style, symbols in [
    ("детское объяснение", Style.CHILDREN, 600),
    ("научный обзор", Style.SCIENTIFIC, 1500),
    ("деловое письмо", Style.OFFICIAL_BUSINESS, 700),
    ("техдокументация", Style.TECHNICAL, 1000),
]:
    features = InputFeaturesService.get_features(label)
    features.input_specifications = Specifications(style_type=style, symbol_length=symbols, word_length=symbols // 7)
    winner = env.get_top_k(features, elements)[0][1].name
    print(f"  {label:<20} -> {winner}")

weights.save_elements(elements)
weights.save_judge(judge)
print()
print(f"Веса и журнал ходов: {db_path.name}")
