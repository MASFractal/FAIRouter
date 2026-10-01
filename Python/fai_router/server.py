"""Сервер, совместимый с OpenAI chat completions, поверх роутера. Через него FAIRouter
подключается к OpenClaw и любому другому клиенту, умеющему говорить с OpenAI: в настройках
указывается адрес сервера и модель fai/auto, а выбор исполнителя делает роутер."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fai_router.router import FaiRouter

# Порт по умолчанию отличается от порта ClawRouter (8402), чтобы не мешать соседям
DEFAULT_PORT = 8412
AUTO_MODEL = "auto"


class RouterHandler(BaseHTTPRequestHandler):
    router: FaiRouter
    lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802 - имя задано http.server
        if self.path.rstrip("/") == "/v1/models":
            models = [AUTO_MODEL, *(candidate.name for candidate in self.router.candidates)]
            self._json(200, {"object": "list", "data": [
                {"id": model, "object": "model", "owned_by": "fai-router"} for model in models]})
            return
        self._json(404, {"error": {"message": f"Нет такого пути: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": f"Нет такого пути: {self.path}"}})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            messages = body.get("messages") or []
            # Роутер не потокобезопасен: ходы идут по одному
            with self.lock:
                answer = self.router.ask_messages(messages)
        except Exception as error:  # noqa: BLE001 - ошибка уходит клиенту в формате OpenAI
            self._json(500, {"error": {"message": str(error), "type": "router_error"}})
            return

        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": answer.winner,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": answer.text}}],
            "usage": {"prompt_tokens": answer.prompt_tokens, "completion_tokens": answer.completion_tokens,
                      "total_tokens": answer.prompt_tokens + answer.completion_tokens},
            "fai_router": {"round_id": answer.round_id, "score": answer.score,
                           "exploration": answer.trace.is_exploration},
        }
        if body.get("stream"):
            self._stream(payload)
        else:
            self._json(200, payload)

    def _stream(self, payload: dict) -> None:
        """Потоковый ответ одним куском: клиенты, просящие stream, получают тот же текст."""
        chunk = {**payload, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": payload["choices"][0]["message"]["content"]},
                              "finish_reason": "stop"}]}
        data = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        print(f"[fai-router] {format % args}")


def make_server(router: FaiRouter, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundRouterHandler", (RouterHandler,), {"router": router})
    return ThreadingHTTPServer((host, port), handler)


def serve(router: FaiRouter, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    server = make_server(router, host, port)
    print(f"FAIRouter слушает http://{host}:{port}/v1, модель для клиентов: {AUTO_MODEL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if router.weights is not None:
            router.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Сервер FAIRouter, совместимый с OpenAI chat completions")
    parser.add_argument("--models", required=True, help="идентификаторы моделей OpenRouter через запятую")
    parser.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY"), help="ключ OpenRouter")
    parser.add_argument("--db", default="fai-router.db", help="файл весов и журнала")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-measure", action="store_true", help="не замерять ответы судьей")
    args = parser.parse_args()
    if not args.key:
        parser.error("нужен ключ: --key или переменная OPENROUTER_API_KEY")

    router = FaiRouter.from_openrouter(args.key, [m.strip() for m in args.models.split(",") if m.strip()],
                                       database_path=args.db, measure=not args.no_measure)
    serve(router, args.host, args.port)


if __name__ == "__main__":
    main()
