"""Production-oriented FastAPI adapter for the shared chat service."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from flaxchat.chat import (
    ChatService,
    GenerationConfig,
    load_chat_service,
    load_chat_service_from_artifact,
)


HTML_PAGE = """<!doctype html><html><head><title>flaxchat</title></head>
<body><main><h1>flaxchat</h1><div id="messages"></div>
<input id="input" maxlength="8000"><button onclick="send()">Send</button></main>
<script>
const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
const messages = document.getElementById('messages'); const input = document.getElementById('input');
function send(){const text=input.value.trim();if(text){ws.send(JSON.stringify({text}));input.value='';}}
ws.onmessage=e=>{const d=JSON.parse(e.data);const p=document.createElement('span');
p.textContent=d.type==='error' ? `[${d.code}] ${d.message}` : (d.text||'');messages.appendChild(p);};
</script></body></html>"""


@dataclass(frozen=True)
class WebSettings:
    model_name: str = "unknown"
    max_input_chars: int = 8_000
    max_tokens: int = 512
    max_concurrent_generations: int = 1
    output_buffer_size: int = 32

    def __post_init__(self) -> None:
        if min(
            self.max_input_chars,
            self.max_tokens,
            self.max_concurrent_generations,
            self.output_buffer_size,
        ) < 1:
            raise ValueError("web service limits must be positive")


def _error(code: str, message: str) -> str:
    return json.dumps({"type": "error", "code": code, "message": message})


def create_app(service: ChatService, settings: WebSettings | None = None) -> FastAPI:
    """Create an injectable app without parsing arguments or loading a model."""
    options = settings or WebSettings()
    app = FastAPI()
    semaphore = asyncio.Semaphore(options.max_concurrent_generations)

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": options.model_name,
            "params": service.model.num_params(),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        cancelled = threading.Event()
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(_error("invalid_json", "message must be valid JSON"))
                    continue
                user_text = message.get("text") if isinstance(message, dict) else None
                if not isinstance(user_text, str) or not user_text.strip():
                    await websocket.send_text(_error("invalid_request", "text must be non-empty"))
                    continue
                if len(user_text) > options.max_input_chars:
                    await websocket.send_text(_error("context_overflow", "input is too long"))
                    continue
                request_text: str = user_text
                if semaphore.locked():
                    await websocket.send_text(_error("overloaded", "generation capacity is busy"))
                    continue
                queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(options.output_buffer_size)
                loop = asyncio.get_running_loop()

                def put(item: tuple[str, str]) -> bool:
                    while not cancelled.is_set():
                        future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                        try:
                            future.result(timeout=0.25)
                            return True
                        except concurrent.futures.TimeoutError:
                            future.cancel()
                    return False

                def generate() -> None:
                    try:
                        config = GenerationConfig(max_tokens=options.max_tokens)
                        for text in service.stream_text(
                            request_text, config, cancelled=cancelled.is_set
                        ):
                            if cancelled.is_set():
                                break
                            if not put(("token", text)):
                                break
                    except ValueError as exc:
                        put(("request_error", str(exc)))
                    except Exception:
                        put(("model_error", "generation failed"))
                    finally:
                        put(("done", ""))

                async with semaphore:
                    worker = asyncio.create_task(asyncio.to_thread(generate))
                    disconnect = asyncio.create_task(websocket.receive())
                    try:
                        while True:
                            output = asyncio.create_task(queue.get())
                            completed, _ = await asyncio.wait(
                                (output, disconnect),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if disconnect in completed:
                                output.cancel()
                                event = disconnect.result()
                                if event["type"] == "websocket.disconnect":
                                    cancelled.set()
                                    await worker
                                    return
                                await websocket.send_text(
                                    _error("overloaded", "wait for generation to finish")
                                )
                                disconnect = asyncio.create_task(websocket.receive())
                                continue
                            kind, text = output.result()
                            if kind == "done":
                                await websocket.send_text(json.dumps({"type": "done"}))
                                break
                            if kind == "token":
                                await websocket.send_text(
                                    json.dumps({"type": "token", "text": text})
                                )
                            else:
                                await websocket.send_text(_error(kind, text))
                        await worker
                    finally:
                        disconnect.cancel()
        except WebSocketDisconnect:
            cancelled.set()
        finally:
            cancelled.set()

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="d12")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument(
        "--artifact-dir",
        help="artifact directory; verifies checksums and loads its manifest",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--checkpoint-type", choices=("base", "sft", "rl"), default="sft")
    parser.add_argument("--max-concurrent-generations", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.artifact_dir and (args.checkpoint_path or args.tokenizer_path):
        parser.error("--artifact-dir cannot be combined with explicit checkpoint paths")
    service = (
        load_chat_service_from_artifact(args.artifact_dir)
        if args.artifact_dir
        else load_chat_service(
            args.model,
            args.checkpoint_type,
            checkpoint_path=args.checkpoint_path,
            tokenizer_path=args.tokenizer_path,
        )
    )
    app = create_app(
        service,
        WebSettings(
            model_name=(
                f"artifact:{Path(args.artifact_dir).name}"
                if args.artifact_dir
                else args.model
            ),
            max_concurrent_generations=args.max_concurrent_generations,
        ),
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
