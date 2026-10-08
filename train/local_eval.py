"""Act 4 on a local NVIDIA GPU: base vs tuned on the eval cases Paper's labels built.

The same scoring as train/eval_models.py (the AI Runtime notebook): each case is judged against
its own guideline by MLflow's ExpectationsGuidelines scorer, which runs on a Databricks-hosted
judge, and the `eval-base-local` and `eval-tuned-local` runs land in the Databricks experiment
for Compare, beside any `eval-base` / `eval-tuned` runs from the AI Runtime job.

    uv run --group local python train/local_eval.py --limit 5     # smoke run
    uv run --group local python train/local_eval.py               # every case
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tapes_eject import config  # noqa: E402

MAX_INPUT_TOKENS = 8192


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tuned", type=Path, default=Path("models/agent_qwen3_4b"))
    p.add_argument("--base-model", default="Qwen/Qwen3-4B")
    p.add_argument("--limit", type=int, default=0, help="0 = every case")
    p.add_argument("--only", choices=["base", "tuned"], help="score just one model")
    args = p.parse_args()

    config.load_dotenv()
    cfg = config.load()
    if args.only != "base" and not args.tuned.exists():
        raise SystemExit(f"{args.tuned} not found: run train/local_sft.py first")

    # One generation at a time: mlflow.genai.evaluate calls predict_fn from a thread pool, and
    # concurrent generate() calls on one GPU run out of memory.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")

    import mlflow
    import torch
    from mlflow.genai.scorers import ExpectationsGuidelines
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mlflow.set_tracking_uri(f"databricks://{cfg.profile}")
    mlflow.set_experiment(cfg.experiment)
    dataset = mlflow.genai.datasets.get_dataset(name=cfg.table("eval_cases"))
    cases = dataset.to_df().head(args.limit) if args.limit else dataset
    print(f"{args.limit or len(dataset.to_df())} eval cases on {torch.cuda.get_device_name(0)}")

    def load(path):
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")

        def predict(messages):
            enc = tok.apply_chat_template(
                list(messages),
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
            enc = {k: v[:, -MAX_INPUT_TOKENS:].to("cuda") for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=512, do_sample=False)
            return tok.decode(out[0][enc["input_ids"].shape[-1] :], skip_special_tokens=True)

        return predict, model

    targets = [("base", args.base_model), ("tuned", str(args.tuned))]
    for name, path in targets:
        if args.only and name != args.only:
            continue
        predict, model = load(path)
        with mlflow.start_run(run_name=f"eval-{name}-local"):
            result = mlflow.genai.evaluate(
                data=cases, predict_fn=predict, scorers=[ExpectationsGuidelines()]
            )
            print(name, result.metrics)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
