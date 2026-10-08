import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tapes_eject.local_gpu import (
    ask_local,
    chat_body,
    chat_reply,
    load_examples,
    run_name,
    sft_kwargs,
)


def test_load_examples_reads_messages_and_enforces_the_minimum(tmp_path):
    path = tmp_path / "training.jsonl"
    rows = [{"session_id": "s1", "messages": [{"role": "user", "content": "hi"}]}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_examples(path, 1) == [{"messages": rows[0]["messages"]}]
    with pytest.raises(SystemExit, match="only 1 training examples"):
        load_examples(path, 20)
    with pytest.raises(SystemExit, match="run `tapes-eject sync` first"):
        load_examples(tmp_path / "missing.jsonl", 1)


def test_sft_kwargs_match_the_notebook():
    lora = sft_kwargs("lora", 5, "/tmp/x")
    full = sft_kwargs("full", 0, "/tmp/x", max_length=2048)
    assert lora["learning_rate"] == 1e-4 and full["learning_rate"] == 1e-5
    assert lora["max_steps"] == 5 and full["max_steps"] == -1  # 0 -> 2 epochs
    assert full["max_length"] == 2048 and lora["bf16"] and lora["report_to"] == "mlflow"
    assert lora["warmup_steps"] == 0.03 and "warmup_ratio" not in lora
    with pytest.raises(ValueError, match="method must be"):
        sft_kwargs("qlora", 5, "/tmp/x")


def test_run_name_marks_local_runs():
    assert run_name("lora", 5) == "sft-lora-5-local"
    assert run_name("full", 0, where="h100") == "sft-full-full-h100"


def test_ask_local_speaks_the_chat_completions_shape():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            data = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "added the flag"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = ask_local(f"http://127.0.0.1:{srv.server_port}/", "Add a --dry-run flag")
    finally:
        srv.shutdown()
    assert out == "added the flag"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"] == chat_body("Add a --dry-run flag")
    assert chat_reply({"choices": [{"message": {"content": "x"}}]}) == "x"
