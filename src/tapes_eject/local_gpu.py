"""The parts of the local-GPU path that need no GPU: reading the training file, the training
settings shared with the AI Runtime notebook, and talking to the local chat server.

Acts 3 and 4 can run on a local NVIDIA GPU (train/local_sft.py, train/local_eval.py,
train/local_serve.py) instead of AI Runtime. Databricks still holds the data, the evaluation
dataset, the experiment, and the scores; only the GPU work moves.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

CHAT_PATH = "/v1/chat/completions"


def load_examples(path: Path, min_examples: int) -> list[dict]:
    """`sync` writes data/training.jsonl; each line is {session_id, messages}."""
    if not path.exists():
        raise SystemExit(f"{path} not found: run `tapes-eject sync` first")
    with open(path, encoding="utf-8") as fh:
        examples = [{"messages": json.loads(line)["messages"]} for line in fh if line.strip()]
    if len(examples) < min_examples:
        raise SystemExit(
            f"only {len(examples)} training examples (need {min_examples}); label more sessions "
            "`golden` in Paper, then `tapes-eject export && tapes-eject sync`, or pass "
            "--min-examples"
        )
    return examples


def sft_kwargs(method: str, max_steps: int, output_dir: str, max_length: int = 4096) -> dict:
    """SFTConfig settings, the same as train/sft_train.py so local and AI Runtime runs compare.
    `max_steps=0` trains 2 epochs."""
    if method not in ("full", "lora"):
        raise ValueError(f"method must be full or lora, not {method!r}")
    return {
        "output_dir": output_dir,
        "max_steps": max_steps if max_steps > 0 else -1,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "learning_rate": 1e-5 if method == "full" else 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0.03,  # a float is a ratio of total steps; warmup_ratio is gone
        "bf16": True,
        "max_length": max_length,
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": 25,
        "save_strategy": "no",
        "report_to": "mlflow",
    }


def run_name(method: str, max_steps: int, where: str = "local") -> str:
    return f"sft-{method}-{max_steps or 'full'}-{where}"


def chat_body(prompt: str) -> dict:
    return {"messages": [{"role": "user", "content": prompt}]}


def chat_reply(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def ask_local(url: str, prompt: str, timeout: int = 600) -> str:
    """One prompt to the local chat server (train/local_serve.py), OpenAI chat-completions shape."""
    req = urllib.request.Request(
        url.rstrip("/") + CHAT_PATH,
        data=json.dumps(chat_body(prompt)).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return chat_reply(json.loads(resp.read().decode("utf-8")))
