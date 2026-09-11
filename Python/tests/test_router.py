import json
import threading
import urllib.request

import numpy as np

from fai_router import FaiRouter, Settings
from fai_router.enums import Capability, Style
from fai_router.routed_element import RoutedElement
from fai_router.server import make_server


class FakeLlm:
    """Подмена модели без сети: отдает готовые JSON по имени схемы. Детские задачи отличает
    по слову в запросе, иначе все задачи журнала совпали бы, вычитание среднего дало бы ноль
    и обучению было бы не от чего двигаться."""

    def complete(self, messages, schema=None, schema_name="answer", **kwargs):
        childish = "ребен" in messages[-1]["content"].lower()
        if schema_name == "input_specifications":
            if childish:
                return json.dumps({"styleType": "Children", "symbolLength": 600, "wordLength": 90,
                                   "paragraphCount": 3, "sectionCount": 0, "listItemCount": 0, "tableCount": 0,
                                   "codeBlockCount": 0, "formulaCount": 0, "headingDepth": 0,
                                   "avgSentenceLength": 8, "readabilityScore": 90, "termDensity": 0.05,
                                   "formalityScore": 0.1, "language": "ru", "hasReferences": False})
            return json.dumps({"styleType": "Scientific", "symbolLength": 1500, "wordLength": 220,
                               "paragraphCount": 4, "sectionCount": 3, "listItemCount": 0, "tableCount": 1,
                               "codeBlockCount": 0, "formulaCount": 0, "headingDepth": 2,
                               "avgSentenceLength": 18, "readabilityScore": 35, "termDensity": 0.7,
                               "formalityScore": 0.9, "language": "ru", "hasReferences": True})
        return json.dumps({"styleType": "Scientific", "termDensity": 0.65, "formalityScore": 0.85})


def make_router(tmp_path, measure=True):
    Settings.llm = FakeLlm()
    shared = np.ones(Settings.full_dim())
    candidates = [RoutedElement("cheap", tps=200, dpmt_inp=0.3, dpmt_outp=2.5, ideal_match_vector=shared.copy()),
                  RoutedElement("strong", tps=60, dpmt_inp=1.0, dpmt_outp=5.0, ideal_match_vector=shared.copy())]
    calls = []

    def execute(candidate, messages):
        calls.append((candidate.name, messages[-1]["content"]))
        return f"# Ответ от {candidate.name}\n\nПервый абзац про кластеризацию. Второй абзац с выводом."

    router = FaiRouter(candidates, execute, database_path=str(tmp_path / "r.db"), measure=measure)
    return router, calls


def test_ask_runs_whole_loop_and_journals(tmp_path):
    router, calls = make_router(tmp_path)
    answer = router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")

    assert answer.winner in ("cheap", "strong")
    assert calls[0][0] == answer.winner
    assert answer.text.startswith("# Ответ от")
    assert answer.requested.style_type == Style.SCIENTIFIC
    assert answer.actual is not None and answer.critic is not None
    assert 0 <= answer.score <= 1
    assert answer.round_id == 1
    assert router.traces.count() == (1, 1)


def test_feedback_train_save_load(tmp_path):
    router, _ = make_router(tmp_path)
    router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    router.ask("Объясни ребенку, что такое кластеризация, в 600 знаков.")
    router.ask("Напиши научный обзор методов регуляризации на 1500 знаков.")
    router.feedback(1, 1.0)
    before = [c.ideal_match_vector.copy() for c in router.candidates]

    loss = router.train(epochs=5)
    assert loss >= 0
    assert Settings.task_mean is not None
    assert any(not np.array_equal(b, c.ideal_match_vector) for b, c in zip(before, router.candidates))

    router.save()
    Settings.task_mean = None
    again = FaiRouter(router.candidates, lambda c, m: "", database_path=str(tmp_path / "r.db"), measure=False)
    assert Settings.task_mean is not None
    # Три хода оценены, но в опыт идет только человеческий отзыв, а он один: автоотзыв
    # температуру не сбивает, разведка не гаснет по мнению собственного судьи
    assert again.candidates[0].experience + again.candidates[1].experience == 1


def test_capability_requirement_reaches_router(tmp_path):
    router, calls = make_router(tmp_path, measure=False)
    router.candidates[0].capabilities = Capability.CODE
    answer = router.ask("Задача с картинкой", required=Capability.VISION)
    assert answer.winner == "strong"


def test_openai_compatible_server(tmp_path):
    router, _ = make_router(tmp_path, measure=False)
    server = make_server(router, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        with urllib.request.urlopen(f"{base}/models") as response:
            models = json.load(response)
        assert [m["id"] for m in models["data"]] == ["auto", "cheap", "strong"]

        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps({"model": "auto", "messages": [
                {"role": "system", "content": "Ты помощник."},
                {"role": "user", "content": "Напиши научный обзор на 1500 знаков."}]}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer anything"})
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
        assert data["object"] == "chat.completion"
        assert data["model"] in ("cheap", "strong")
        assert data["choices"][0]["message"]["content"].startswith("# Ответ от")
        assert data["fai_router"]["round_id"] == 1

        streamed = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps({"model": "auto", "stream": True,
                             "messages": [{"role": "user", "content": "Еще раз на 1500 знаков."}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(streamed) as response:
            text = response.read().decode()
        assert text.startswith("data: ") and text.rstrip().endswith("data: [DONE]")
    finally:
        server.shutdown()
        server.server_close()


def test_len_answer_follows_recognized_order(tmp_path):
    """Объем ответа берется из распознанного заказа, а не из длины промпта: короткая просьба о
    длинном тексте иначе выглядела дешевой и быстрой у всех кандидатов разом."""
    from fai_router.services import InputFeaturesService

    make_router(tmp_path, measure=False)
    short_prompt = "Обзор на 1500 знаков."
    features = InputFeaturesService.get_features_full(short_prompt)
    assert features.input_specifications.symbol_length == 1500
    assert features.len_answer == 1500 / InputFeaturesService.EST_SYMBOL_PER_TOKEN
    assert features.len_answer > InputFeaturesService.get_features(short_prompt).len_answer


def test_judge_learns_only_from_human_feedback(tmp_path):
    """После хода стоит автоотзыв; обучение по нему не должно трогать судью, иначе одна
    автоматическая оценка подгонялась бы под другую. Человеческий отзыв судью двигает."""
    router, _ = make_router(tmp_path)
    router.ask("Напиши научный обзор методов кластеризации на 1500 знаков.")
    before = router.judge.transformer_w.copy()
    router.train()
    assert np.array_equal(before, router.judge.transformer_w)

    router.feedback(1, 0.0)
    router.train()
    assert not np.array_equal(before, router.judge.transformer_w)
