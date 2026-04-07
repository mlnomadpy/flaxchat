"""
Web chat UI using FastAPI + WebSocket streaming.

Usage:
    python -m scripts.chat_web --model=d12 --port=8000
"""

import argparse
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from flax import nnx

from flaxchat.gpt import GPT
from flaxchat.config import FlaxChatConfig
from flaxchat.common import get_base_dir, print0
from flaxchat.tokenizer import get_tokenizer
from flaxchat.engine import generate_with_cache
from flaxchat.checkpoint import restore_model_from_checkpoint

parser = argparse.ArgumentParser(description="Web Chat UI")
parser.add_argument("--model", type=str, default="d12")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--checkpoint-type", type=str, default="sft", choices=["base", "sft"])
args = parser.parse_args()

# Load model
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

depth = int(args.model.replace("d", ""))
config = FlaxChatConfig.from_depth(depth=depth, vocab_size=vocab_size)
model = GPT(config.model, rngs=nnx.Rngs(0))

base_dir = get_base_dir()
ckpt_dir = f"{args.checkpoint_type}_checkpoints"
checkpoint_dir = f"{base_dir}/{ckpt_dir}/{args.model}"
print0(f"Loading model from {checkpoint_dir}")
restore_model_from_checkpoint(model, checkpoint_dir)
print0(f"Model loaded: {model.num_params():,} params")

app = FastAPI()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>flaxchat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a2e; color: #eee; }
        .container { max-width: 800px; margin: 0 auto; height: 100vh; display: flex; flex-direction: column; }
        .header { padding: 20px; text-align: center; border-bottom: 1px solid #333; }
        .header h1 { font-size: 1.5em; color: #7c3aed; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; }
        .message { margin: 10px 0; padding: 12px 16px; border-radius: 12px; max-width: 80%; }
        .user { background: #7c3aed; margin-left: auto; }
        .assistant { background: #2a2a4a; }
        .input-area { padding: 20px; border-top: 1px solid #333; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #444; background: #2a2a4a; color: #eee; font-size: 16px; }
        .input-area button { padding: 12px 24px; border-radius: 8px; border: none; background: #7c3aed; color: #fff; cursor: pointer; font-size: 16px; }
        .input-area button:hover { background: #6d28d9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>flaxchat</h1></div>
        <div class="messages" id="messages"></div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Type a message..." autofocus />
            <button onclick="send()">Send</button>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        const messages = document.getElementById('messages');
        const input = document.getElementById('input');
        let currentAssistant = null;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'token') {
                if (!currentAssistant) {
                    currentAssistant = document.createElement('div');
                    currentAssistant.className = 'message assistant';
                    messages.appendChild(currentAssistant);
                }
                currentAssistant.textContent += data.text;
                messages.scrollTop = messages.scrollHeight;
            } else if (data.type === 'done') {
                currentAssistant = null;
            }
        };

        function send() {
            const text = input.value.trim();
            if (!text) return;
            const msg = document.createElement('div');
            msg.className = 'message user';
            msg.textContent = text;
            messages.appendChild(msg);
            ws.send(JSON.stringify({text: text}));
            input.value = '';
            messages.scrollTop = messages.scrollHeight;
        }

        input.addEventListener('keypress', (e) => { if (e.key === 'Enter') send(); });
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(HTML_PAGE)

@app.get("/health")
async def health():
    return {"status": "ok", "model": args.model, "params": model.num_params()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_text = msg.get("text", "")

            # Rate limiting
            if len(user_text) > 8000:
                user_text = user_text[:8000]

            # Build tokens
            bos = tokenizer.encode_special("<|bos|>")
            user_start = tokenizer.encode_special("<|user_start|>")
            user_end = tokenizer.encode_special("<|user_end|>")
            assistant_start = tokenizer.encode_special("<|assistant_start|>")
            assistant_end = tokenizer.encode_special("<|assistant_end|>")

            tokens = [bos, user_start] + tokenizer.encode(user_text) + [user_end, assistant_start]

            # Generate
            output = generate_with_cache(model, tokens, max_tokens=512, temperature=0.8, top_k=50)

            # Stream tokens after prompt
            response_tokens = output[len(tokens):]
            for t in response_tokens:
                if t == assistant_end:
                    break
                text = tokenizer.decode([t])
                await websocket.send_text(json.dumps({"type": "token", "text": text}))

            await websocket.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
