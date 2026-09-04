import importlib

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
