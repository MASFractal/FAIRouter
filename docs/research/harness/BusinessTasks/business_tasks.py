"""Стенд бизнес-задач: три замера для документа docs/research/business-evaluation.md.

1. Распознавание: верно ли модель распознавания определяет тип задачи, область, экспертность и
   смысловые пункты на 66 задачах набора data/business-tasks.json.
2. Живой прогон: 16 задач (по две на группу) на четырех моделях; форма, содержание и итог по
   моделям и группам, и совпадает ли лучший по содержанию с тем, кого выбрал бы прайор рейтингов.
3. Контрольные пары: ответ одной формы, хороший и плохой по содержанию. Форма обязана их не
   различить, судья содержания обязан развести.

Ответы моделей и оценки кэшируются в results/cache.json: повторный прогон бесплатен, а новые
задачи или модели дозапрашиваются. Ключ OpenRouter берется из OPENROUTER_API_KEY или из key.txt
рядом со стендом либо у стенда RouterEval.

    python business_tasks.py [--limit N] [--only recognition|live|controls]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "Python"))

from fai_router import Judge  # noqa: E402
from fai_router.benchmarks import BenchmarkSnapshot  # noqa: E402
from fai_router.llm.client import OpenRouterClient  # noqa: E402
from fai_router.llm.content_judge import ContentJudge  # noqa: E402
from fai_router.llm.spec_input import SpecInputRecognizer  # noqa: E402
from fai_router.services import InputFeaturesService, SpecOutputService  # noqa: E402
from fai_router.settings import Settings  # noqa: E402
from fai_router.specifications import Specifications  # noqa: E402
from fai_router.tracking import InputFeatures  # noqa: E402
from fai_router.training import benchmark_prior  # noqa: E402

TASKS = ROOT / "data" / "business-tasks.json"
SNAPSHOT = ROOT / "data" / "benchmark-snapshot.json"
CONTROLS = HERE / "control-pairs.json"
RESULTS = HERE / "results"
CACHE = RESULTS / "cache.json"

RECOGNIZER_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-5.2"
CANDIDATES = ["google/gemini-2.5-flash", "openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5", "deepseek/deepseek-v4-flash"]

GROUPS = {
    "Письма и коммуникации": {"BusinessLetter", "CommercialOffer", "CustomerReply", "InternalMemo", "EmailCampaign"},
    "Отчеты и аналитика": {"AnalyticalReport", "FinancialReport", "MarketResearch", "DataAnalysis", "Summary", "MeetingMinutes"},
    "Маркетинг и продажи": {"MarketingArticle", "SocialPost", "AdCopy", "ProductDescription", "LandingPage", "PressRelease", "SalesScript"},
    "Документы и право": {"LegalDocument", "LegalAnalysis", "Policy", "JobDescription"},
    "Код и ИТ": {"CodeWriting", "CodeReview", "TechnicalDoc"},
    "Наука и обучение": {"Explanation", "ResearchReview", "LearningMaterial", "MathSolution"},
    "Творчество и медиа": {"Story", "Script"},
    "Консультации и планы": {"Advice", "BusinessPlan"},
}

_lock = threading.Lock()

# Метка кэша распознанных заданий и их оценок; задается ключом --spec-tag
SPEC_TAG = ""


def api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"].strip()
    for path in (HERE / "key.txt", HERE.parent / "RouterEval" / "key.txt"):
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    sys.exit("Нет ключа: задайте OPENROUTER_API_KEY или положите key.txt рядом со стендом.")


def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def save_cache(cache: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with _lock:
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def cached(cache: dict, key: str, compute):
    with _lock:
        if key in cache:
            return cache[key]
    value = compute()
    with _lock:
        cache[key] = value
    return value


def group_of(kind: str) -> str:
    return next(name for name, kinds in GROUPS.items() if kind in kinds)


def recognize(cache: dict, recognizer: SpecInputRecognizer, prompt: str) -> Specifications:
    return Specifications.from_dict(cached(cache, f"spec{SPEC_TAG}:" + prompt, lambda: recognizer.get_specifications(prompt).to_dict()))


def assess(cache: dict, key: str, prompt: str, order: Specifications, answer: str,
           measurer: SpecOutputService, judge: ContentJudge) -> dict:
    """Форма, содержание и итог ответа; кэшируются по ключу."""

    def compute() -> dict:
        actual = measurer.get_specifications(answer)
        content = judge.review(prompt, order, answer)
        critic = Judge.criticize(order, actual, content)
        return {
            "form": 1 - critic.form_deviation,
            "content": content.score,
            "assessment": Judge.assess(critic, content),
            "criteria": {item.name: item.score for item in content.criteria},
            "claims": [{"text": claim.text, "truth": claim.truth} for claim in content.claims],
            "issues": content.issues,
            "form_mismatches": [item.field for item in critic.mismatches if not item.content],
            "mismatches": [f"{item.field}: заказано {item.requested}, получено {item.actual}"
                           for item in critic.mismatches if item.content],
        }

    return cached(cache, key, compute)


def measure_recognition(cache: dict, tasks: list[dict], recognizer: SpecInputRecognizer) -> None:
    with ThreadPoolExecutor(8) as pool:
        specs = list(pool.map(lambda task: recognize(cache, recognizer, task["prompt"]), tasks))
    save_cache(cache)

    kind_hits = sum(spec.task_kind.value == task["kind"] for spec, task in zip(specs, tasks))
    domain_hits = sum(spec.domain.value == task["domain"] for spec, task in zip(specs, tasks))
    group_hits = sum(spec.task_kind.value in GROUPS[group_of(task["kind"])] for spec, task in zip(specs, tasks))
    expert_hits = sum((spec.expert_level >= 0.6) == (task["expert"] == "high") for spec, task in zip(specs, tasks))
    with_points = sum(bool(spec.required_points) for spec in specs)
    constraints = sum(bool(spec.constraints) for spec in specs)
    n = len(tasks)
    print("\n## Распознавание\n")
    print(f"| Величина | Значение |\n|---|---|")
    print(f"| Задач | {n} |")
    print(f"| Тип задачи угадан точно | {kind_hits} из {n} ({kind_hits / n:.0%}) |")
    print(f"| Тип задачи попал в свою группу | {group_hits} из {n} ({group_hits / n:.0%}) |")
    print(f"| Область угадана | {domain_hits} из {n} ({domain_hits / n:.0%}) |")
    print(f"| Экспертность в своей половине шкалы | {expert_hits} из {n} ({expert_hits / n:.0%}) |")
    print(f"| Выделены смысловые пункты | {with_points} из {n} |")
    print(f"| Выделены явные ограничения | {constraints} из {n} |")
    misses = [(task["id"], task["kind"], spec.task_kind.value) for spec, task in zip(specs, tasks)
              if spec.task_kind.value != task["kind"]]
    print("\nПромахи типа задачи: " + "; ".join(f"{i} {want} -> {got}" for i, want, got in misses))
    domain_misses = [(task["id"], task["domain"], spec.domain.value) for spec, task in zip(specs, tasks)
                     if spec.domain.value != task["domain"]]
    print("\nПромахи области: " + "; ".join(f"{i} {want} -> {got}" for i, want, got in domain_misses))


def measure_live(cache: dict, tasks: list[dict], recognizer: SpecInputRecognizer, key: str,
                 measurer: SpecOutputService, judge: ContentJudge) -> None:
    live = [task for task in tasks if task.get("live")]
    clients = {model: OpenRouterClient(key, model) for model in CANDIDATES}

    def answer(task: dict, model: str) -> str:
        return cached(cache, f"answer:{model}:{task['id']}",
                      lambda: clients[model].complete([{"role": "user", "content": task["prompt"]}]))

    def run(pair):
        task, model = pair
        order = recognize(cache, recognizer, task["prompt"])
        text = answer(task, model)
        return task, model, assess(cache, f"assess{SPEC_TAG}:{model}:{task['id']}", task["prompt"], order, text, measurer, judge)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(run, [(task, model) for task in live for model in CANDIDATES]))
    save_cache(cache)

    snapshot = BenchmarkSnapshot.load(str(SNAPSHOT)) if SNAPSHOT.exists() else None
    vectors = {model: benchmark_prior.vector(snapshot, model) if snapshot else None for model in CANDIDATES}
    short = {model: model.split("/")[1] for model in CANDIDATES}

    print("\n## Живой прогон: средние по моделям\n")
    print("| Модель | Форма | Содержание | Итог |\n|---|---|---|---|")
    for model in CANDIDATES:
        mine = [r for t, m, r in rows if m == model]
        print(f"| {short[model]} | {avg(r['form'] for r in mine):.2f} | {avg(r['content'] for r in mine):.2f} | "
              f"{avg(r['assessment'] for r in mine):.2f} |")

    print("\n## Живой прогон: содержание по группам\n")
    print("| Группа | " + " | ".join(short[m] for m in CANDIDATES) + " | Лучший по содержанию | Лучший по форме | Выбор прайора |")
    print("|---|" + "---|" * (len(CANDIDATES) + 3))
    agree_content = agree_form_content = 0
    for group in GROUPS:
        in_group = [(t, m, r) for t, m, r in rows if group_of(t["kind"]) == group]
        if not in_group:
            continue
        content = {m: avg(r["content"] for t, mm, r in in_group if mm == m) for m in CANDIDATES}
        form = {m: avg(r["form"] for t, mm, r in in_group if mm == m) for m in CANDIDATES}
        best_content = max(content, key=content.get)
        best_form = max(form, key=form.get)
        prior_pick = prior_choice(in_group, vectors, recognizer, cache)
        agree_content += prior_pick == best_content
        agree_form_content += best_form == best_content
        print(f"| {group} | " + " | ".join(f"{content[m]:.2f}" for m in CANDIDATES)
              + f" | {short[best_content]} | {short[best_form]} | {short.get(prior_pick, 'нет данных')} |")
    print(f"\nЛучший по форме совпал с лучшим по содержанию в {agree_form_content} группах из {len(GROUPS)}; "
          f"выбор прайора совпал с лучшим по содержанию в {agree_content}.")

    print("\n## Живой прогон: критерии содержания, среднее по всем ответам\n")
    names = list(rows[0][2]["criteria"])
    print("| Критерий | " + " | ".join(short[m] for m in CANDIDATES) + " |\n|---|" + "---|" * len(CANDIDATES))
    for name in names:
        cells = []
        for model in CANDIDATES:
            values = [r["criteria"][name] for t, m, r in rows if m == model and r["criteria"][name] is not None]
            cells.append(f"{avg(values):.2f} ({len(values)})" if values else "нет")
        print(f"| {name} | " + " | ".join(cells) + " |")


def prior_choice(in_group, vectors, recognizer, cache) -> str | None:
    """Кого выбрал бы прайор рейтингов по средней оценке на задачах группы."""
    if not any(v is not None for v in vectors.values()):
        return None
    Settings.task_mean = None
    tasks = {t["id"]: t for t, m, r in in_group}.values()
    scores = {}
    for model, vector in vectors.items():
        if vector is None:
            continue
        values = []
        for task in tasks:
            order = recognize(cache, recognizer, task["prompt"])
            features = InputFeatures(len(task["prompt"]) / InputFeaturesService.EST_SYMBOL_PER_TOKEN,
                                     max(order.symbol_length, 600) / InputFeaturesService.EST_SYMBOL_PER_TOKEN, order)
            values.append(float(features.feature_vector() @ vector))
        scores[model] = avg(values)
    return max(scores, key=scores.get) if scores else None


def measure_controls(cache: dict, recognizer: SpecInputRecognizer, measurer: SpecOutputService, judge: ContentJudge) -> None:
    pairs = json.loads(CONTROLS.read_text(encoding="utf-8"))["pairs"]

    def run(item):
        pair, side = item
        order = recognize(cache, recognizer, pair["prompt"])
        return pair, side, assess(cache, f"control{SPEC_TAG}:{pair['id']}:{side}", pair["prompt"], order, pair[side], measurer, judge)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(run, [(pair, side) for pair in pairs for side in ("good", "bad")]))
    save_cache(cache)

    print("\n## Контрольные пары\n")
    print("| Пара | Изъян плохого ответа | Форма: хороший / плохой | Содержание: хороший / плохой | Итог: хороший / плохой |")
    print("|---|---|---|---|---|")
    by = {(p["id"], side): r for p, side, r in rows}
    for pair in pairs:
        good, bad = by[(pair["id"], "good")], by[(pair["id"], "bad")]
        print(f"| {pair['id']} | {pair['defect']} | {good['form']:.2f} / {bad['form']:.2f} | "
              f"{good['content']:.2f} / {bad['content']:.2f} | {good['assessment']:.2f} / {bad['assessment']:.2f} |")
    for pair in pairs:
        bad = by[(pair["id"], "bad")]
        print(f"\nРасхождения с ТЗ у плохого ответа «{pair['id']}» ({len(bad['mismatches'])}): "
              + "; ".join(bad["mismatches"][:8]))
        print(f"Расхождения с ТЗ у хорошего ответа «{pair['id']}» ({len(by[(pair['id'], 'good')]['mismatches'])}): "
              + "; ".join(by[(pair['id'], 'good')]['mismatches'][:8]))


def avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="сколько задач брать (для пробного прогона)")
    parser.add_argument("--only", choices=["recognition", "live", "controls"])
    parser.add_argument("--spec-tag", default="", help="метка кэша заданий: новый замер распознавания рядом с прежним")
    args = parser.parse_args()
    global SPEC_TAG
    SPEC_TAG = args.spec_tag

    key = api_key()
    cache = load_cache()
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]

    recognition_llm = OpenRouterClient(key, RECOGNIZER_MODEL)
    recognizer = SpecInputRecognizer(recognition_llm)
    measurer = SpecOutputService(recognition_llm)
    judge = ContentJudge(OpenRouterClient(key, JUDGE_MODEL))

    print(f"Распознавание и замер формы: {RECOGNIZER_MODEL}; судья содержания: {JUDGE_MODEL}; "
          f"кандидаты: {', '.join(CANDIDATES)}")
    try:
        if args.only in (None, "recognition"):
            measure_recognition(cache, tasks, recognizer)
        if args.only in (None, "controls"):
            measure_controls(cache, recognizer, measurer, judge)
        if args.only in (None, "live"):
            measure_live(cache, tasks, recognizer, key, measurer, judge)
    finally:
        save_cache(cache)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
