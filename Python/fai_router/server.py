"""Сервер, совместимый с OpenAI chat completions, поверх роутера. Через него FAIRouter
подключается к OpenClaw и любому другому клиенту, умеющему говорить с OpenAI: в настройках
указывается адрес сервера и модель fai/auto, а выбор исполнителя делает роутер.

Пути: GET /v1/models, POST /v1/chat/completions и POST /v1/feedback (отзыв на ход: {"round_id": 1,
"score": 0.9}). С токеном (довод token или переменная FAI_ROUTER_TOKEN) каждый запрос обязан нести
заголовок Authorization: Bearer <токен>. Сервер на внешнем адресе без токена отвечает любому, кто
до него дотянется, и тратит ключ поставщика: об этом он предупреждает при запуске."""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from fai_router.llm.client import OPENROUTER_URL, base_url_for_key
from fai_router.router import FaiRouter

log = logging.getLogger("fai_router")

# Порт по умолчанию отличается от порта ClawRouter (8402), чтобы не мешать соседям
DEFAULT_PORT = 8412
AUTO_MODEL = "auto"

# Переменная окружения с токеном доступа к серверу
TOKEN_VARIABLE = "FAI_ROUTER_TOKEN"

# Наибольший размер тела запроса: диалог с длинной историей укладывается, а гигабайт в память не читается
MAX_BODY_BYTES = 8 * 1024 * 1024


class RouterHandler(BaseHTTPRequestHandler):
    router: FaiRouter
    token: str | None = None
    max_body: int = MAX_BODY_BYTES

    def do_GET(self) -> None:  # noqa: N802 - имя задано http.server
        if not self._authorized():
            return
        if self.path.rstrip("/") == "/v1/models":
            models = [AUTO_MODEL, *(candidate.name for candidate in self.router.candidates)]
            self._json(200, {"object": "list", "data": [
                {"id": model, "object": "model", "owned_by": "fai-router"} for model in models]})
            return
        self._error(404, f"Нет такого пути: {self.path}", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/feedback"):
            self._error(404, f"Нет такого пути: {self.path}", "not_found")
            return
        body = self._body()
        if body is None:
            return
        if path == "/v1/feedback":
            self._feedback(body)
        else:
            self._complete(body)

    def _complete(self, body: dict[str, Any]) -> None:
        # Роутер не держит общей блокировки на ход: выбор читает векторы, а обучение подменяет их
        # целиком под своей блокировкой, поэтому ходы идут параллельно
        try:
            answer = self.router.ask_messages(body.get("messages") or [],
                                              max_tokens=_positive_int(body.get("max_tokens", body.get("max_completion_tokens"))),
                                              temperature=_number(body.get("temperature")))
        except ValueError as error:
            self._error(400, str(error), "invalid_request_error")
            return
        except Exception as error:  # noqa: BLE001 - ошибка уходит клиенту в формате OpenAI
            log.warning("Ход не выполнен (%s).", type(error).__name__)
            self._error(500, f"Ход не выполнен: {type(error).__name__}", "router_error")
            return

        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": answer.winner,
            "choices": [{"index": 0, "finish_reason": answer.finish_reason,
                         "message": {"role": "assistant", "content": answer.text}}],
            "usage": {"prompt_tokens": answer.prompt_tokens, "completion_tokens": answer.completion_tokens,
                      "total_tokens": answer.prompt_tokens + answer.completion_tokens},
            "fai_router": {"round_id": answer.round_id, "score": answer.score, "assessment": answer.assessment,
                           "exploration": answer.trace.is_exploration, "reached": answer.reached,
                           "failed": answer.trace.failed, "context_shortfall": answer.trace.context_shortfall},
        }
        if body.get("stream"):
            self._stream(payload)
        else:
            self._json(200, payload)

    def _feedback(self, body: dict[str, Any]) -> None:
        """Отзыв на ход: round_id из ответа (поле fai_router.round_id) и score от 0 до 1; human
        по умолчанию истина."""
        round_id, score = body.get("round_id"), _number(body.get("score"))
        if not isinstance(round_id, int) or isinstance(round_id, bool) or score is None:
            self._error(400, "Нужны round_id (целое) и score (число от 0 до 1).", "invalid_request_error")
            return
        try:
            self.router.feedback(round_id, score, human=bool(body.get("human", True)))
        except (ValueError, RuntimeError) as error:
            self._error(400, str(error), "invalid_request_error")
            return
        self._json(200, {"ok": True, "round_id": round_id})

    def _authorized(self) -> bool:
        if not self.token:
            return True
        header = self.headers.get("Authorization", "")
        given = header[7:].strip() if header[:7].lower() == "bearer " else ""
        if hmac.compare_digest(given.encode("utf-8"), self.token.encode("utf-8")):
            return True
        self._error(401, "Нужен заголовок Authorization: Bearer <токен сервера>.", "authentication_error")
        return False

    def _body(self) -> dict[str, Any] | None:
        """Тело запроса JSON-объектом; None, если ответ об ошибке уже отправлен."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self._error(400, "Неверный Content-Length.", "invalid_request_error")
            return None
        if length > self.max_body:
            self._error(413, f"Тело запроса больше {self.max_body} байт.", "invalid_request_error")
            self.close_connection = True
            return None
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict):
            self._error(400, "Тело запроса должно быть JSON-объектом.", "invalid_request_error")
            return None
        return body

    def _stream(self, payload: dict) -> None:
        """Потоковый ответ одним куском: клиенты, просящие stream, получают тот же текст."""
        chunk = {**payload, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": payload["choices"][0]["message"]["content"]},
                              "finish_reason": payload["choices"][0]["finish_reason"]}]}
        data = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str) -> None:
        self._json(status, {"error": {"message": message, "type": kind}})

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        print(f"[fai-router] {format % args}")


def make_server(router: FaiRouter, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                token: str | None = None, max_body: int = MAX_BODY_BYTES) -> ThreadingHTTPServer:
    """Сервер над роутером. Токен не задан, тогда берется из FAI_ROUTER_TOKEN; пустой токен значит
    «без проверки», и на внешнем адресе это предупреждение."""
    token = token if token is not None else os.environ.get(TOKEN_VARIABLE) or None
    if not token and not _loopback(host):
        log.warning("Сервер слушает %s без токена: любой, кто до него дотянется, тратит ключ поставщика. "
                    "Задайте --token или переменную %s.", host, TOKEN_VARIABLE)
    handler = type("BoundRouterHandler", (RouterHandler,), {"router": router, "token": token, "max_body": max_body})
    return ThreadingHTTPServer((host, port), handler)


def serve(router: FaiRouter, host: str = "127.0.0.1", port: int = DEFAULT_PORT, token: str | None = None,
          train_every: float | None = None, reload_every: float | None = None) -> None:
    """Запуск сервера. train_every: раз в столько секунд роутер учится на новых отзывах и сохраняет
    веса. reload_every: раз в столько секунд веса перечитываются из базы, если их учит другой процесс
    (учить базу стоит одному процессу: двое затирали бы веса друг друга). При остановке веса
    сохраняются, только если этот процесс что-то обучил и не сохранил: иначе save затер бы веса,
    обученные другим процессом."""
    server = make_server(router, host, port, token)
    stop = threading.Event()
    if train_every or reload_every:
        threading.Thread(target=_maintain, args=(router, train_every, reload_every, stop), daemon=True).start()
    print(f"FAIRouter слушает http://{host}:{port}/v1, модель для клиентов: {AUTO_MODEL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        if router.unsaved:
            router.save()


def _maintain(router: FaiRouter, train_every: float | None, reload_every: float | None,
              stop: threading.Event) -> None:
    """Периодическое обучение с сохранением либо перечитывание весов."""
    interval = train_every or reload_every
    while not stop.wait(interval):
        try:
            if train_every:
                if router.train().rounds:
                    router.save()
            else:
                router.load()
        except Exception as error:  # noqa: BLE001 - сбой обучения сервер не роняет
            log.warning("Обслуживание весов не удалось (%s).", type(error).__name__)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _positive_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _parse_prices(items: list[str]) -> dict[str, tuple[float, float]]:
    """Цены из командной строки: модель=вход:выход, за миллион токенов в валюте поставщика."""
    prices = {}
    for item in items:
        try:
            model, pair = item.split("=", 1)
            inp, outp = pair.split(":", 1)
            prices[model.strip()] = (float(inp), float(outp))
        except ValueError as error:
            raise SystemExit(f"Цена задается как модель=вход:выход, получено: {item}") from error
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Сервер FAIRouter, совместимый с OpenAI chat completions")
    parser.add_argument("--models", default="popular",
                        help="идентификаторы моделей через запятую либо набор: popular (по умолчанию) или all")
    parser.add_argument("--profile", choices=["quality", "balance", "price"], default="balance",
                        help="профиль весов: качество прежде всего, баланс (по умолчанию) или экономный")
    parser.add_argument("--bar", type=float, default=None, metavar="0..1",
                        help="планка достаточности: обязательная вероятность лайка; калибруется по журналу "
                             "человеческих отзывов, до их накопления выбор идет без планки")
    parser.add_argument("--provider", choices=["auto", "fractalrouter", "openrouter"], default="auto",
                        help="поставщик: FractalRouter или OpenRouter; auto опознает его по виду ключа")
    parser.add_argument("--base-url", default=None,
                        help="адрес любого поставщика по протоколу OpenAI, вида https://host/v1")
    parser.add_argument("--key", default=os.environ.get("FRACTALROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
                        help="ключ поставщика; по умолчанию FRACTALROUTER_API_KEY либо OPENROUTER_API_KEY")
    parser.add_argument("--price", action="append", default=[], metavar="МОДЕЛЬ=ВХОД:ВЫХОД",
                        help="цена модели за миллион токенов в валюте поставщика; без нее цена берется из его каталога")
    parser.add_argument("--db", default="fai-router.db", help="файл весов и журнала")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=None,
                        help=f"токен доступа к серверу (заголовок Authorization: Bearer); по умолчанию {TOKEN_VARIABLE}")
    parser.add_argument("--train-every", type=float, default=None, metavar="СЕКУНД",
                        help="учиться на новых отзывах и сохранять веса раз в столько секунд")
    parser.add_argument("--reload-every", type=float, default=None, metavar="СЕКУНД",
                        help="перечитывать веса из базы раз в столько секунд (их учит другой процесс)")
    parser.add_argument("--no-measure", action="store_true", help="не замерять ответы судьей")
    args = parser.parse_args()
    if not args.key:
        parser.error("нужен ключ: --key либо переменная FRACTALROUTER_API_KEY или OPENROUTER_API_KEY")

    models = args.models.strip() if args.models.strip().lower() in ("all", "popular") \
        else [m.strip() for m in args.models.split(",") if m.strip()]
    options = dict(database_path=args.db, prices=_parse_prices(args.price), measure=not args.no_measure,
                   weights=args.profile, bar=args.bar)
    if args.base_url:
        router = FaiRouter.from_openai_compatible(args.base_url, args.key, models, **options)
    elif args.provider == "openrouter" or (args.provider == "auto" and _provider(parser, args.key) == OPENROUTER_URL):
        router = FaiRouter.from_openai_compatible(OPENROUTER_URL, args.key, models, **options)
    else:
        router = FaiRouter.from_fractalrouter(args.key, models, **options)
    serve(router, args.host, args.port, args.token, args.train_every, args.reload_every)


def _provider(parser: argparse.ArgumentParser, key: str) -> str:
    try:
        return base_url_for_key(key)
    except ValueError as error:
        parser.error(f"{error} Либо укажите --provider.")
        raise


if __name__ == "__main__":
    main()
