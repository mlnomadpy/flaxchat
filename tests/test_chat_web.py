import importlib
import threading
import time

import pytest
from fastapi.testclient import TestClient

from scripts.chat_web import WebSettings, create_app


class _Model:
    def num_params(self):
        return 7


class _Service:
    model = _Model()

    def stream_text(self, text, config, *, cancelled=None):
        del config
        for token in (text, "!"):
            if cancelled and cancelled():
                return
            yield token


class _CancellableService:
    model = _Model()

    def __init__(self):
        self.cancelled = threading.Event()

    def stream_text(self, text, config, *, cancelled=None):
        del text, config
        while not cancelled():
            time.sleep(0.01)
        self.cancelled.set()
        if False:
            yield ""


class _FailingService:
    model = _Model()

    def stream_text(self, text, config, *, cancelled=None):
        del text, config, cancelled
        raise RuntimeError("private model detail")
        yield


def test_import_has_no_model_loading(monkeypatch):
    monkeypatch.setattr("flaxchat.chat.load_chat_service", lambda *_a, **_k: 1 / 0)
    importlib.reload(importlib.import_module("scripts.chat_web"))


def test_application_factory_health_and_websocket_protocol():
    app = create_app(_Service(), WebSettings(model_name="tiny", max_tokens=2))
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok", "model": "tiny", "params": 7
        }
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text("not-json")
            assert websocket.receive_json()["code"] == "invalid_json"
            websocket.send_json({"text": "hi"})
            assert websocket.receive_json() == {"type": "token", "text": "hi"}
            assert websocket.receive_json() == {"type": "token", "text": "!"}
            assert websocket.receive_json() == {"type": "done"}


def test_websocket_rejects_oversized_input():
    app = create_app(_Service(), WebSettings(max_input_chars=2))
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.send_json({"text": "too long"})
        assert websocket.receive_json()["code"] == "context_overflow"


def test_websocket_disconnect_cancels_silent_generation():
    service = _CancellableService()
    app = create_app(service)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"text": "start"})
        assert service.cancelled.wait(timeout=1)


def test_websocket_returns_sanitized_model_error():
    app = create_app(_FailingService())
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.send_json({"text": "start"})
        assert websocket.receive_json() == {
            "type": "error",
            "code": "model_error",
            "message": "generation failed",
        }
        assert websocket.receive_json() == {"type": "done"}


def test_web_settings_reject_nonpositive_bounds():
    with pytest.raises(ValueError, match="must be positive"):
        WebSettings(output_buffer_size=0)
