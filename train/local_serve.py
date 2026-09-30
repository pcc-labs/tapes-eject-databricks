"""The tuned model as a local HTTP endpoint, in place of Model Serving.

One route, POST /v1/chat/completions, in the OpenAI chat-completions shape that
`tapes-eject ask --url http://127.0.0.1:8081 "<prompt>"` sends. One request at a time; this is
a demo server on one GPU, not a production one.

    uv run --group local python train/local_serve.py                 # models/agent_qwen3_4b
    uv run --group local python train/local_serve.py --model Qwen/Qwen3-4B   # the base, to compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tapes_eject.local_gpu import CHAT_PATH  # noqa: E402

MAX_INPUT_TOKENS = 8192


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="models/agent_qwen3_4b")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    served = Path(args.model).name

    def generate(messages: list[dict]) -> str:
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v[:, -MAX_INPUT_TOKENS:].to("cuda") for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[-1] :], skip_special_tokens=True)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/ping":
                self._json(200, {"status": "ok", "model": served})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != CHAT_PATH:
                self._json(404, {"error": f"POST {CHAT_PATH}"})
                return
            n = int(self.headers.get("content-length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                messages = body["messages"]
            except (ValueError, KeyError):
                self._json(422, {"error": "body must be {messages: [{role, content}, ...]}"})
                return
            text = generate(messages)
            self._json(
                200,
                {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "model": served,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        def log_message(self, fmt, *a):  # quieter than the default
            print(f"{self.command} {self.path} {a[1] if len(a) > 1 else ''}", flush=True)

    print(f"serving {served} on http://{args.host}:{args.port}{CHAT_PATH}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
