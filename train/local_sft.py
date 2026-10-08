"""Act 3 on a local NVIDIA GPU: SFT of Qwen3-4B on the examples Paper's labels selected.

The same data, hyperparameters, and run naming as train/sft_train.py (the AI Runtime notebook),
minus dbutils and the Unity Catalog model registration. The run logs to the Databricks MLflow
experiment, so it sits beside any AI Runtime run. LoRA is the default: a full fine-tune of a 4B
model needs about 48 GB with optimizer states, more than a 32 GB card holds.

    uv run --group local python train/local_sft.py --max-steps 5     # smoke run
    uv run --group local python train/local_sft.py --max-steps 0     # 2 epochs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tapes_eject import config  # noqa: E402
from tapes_eject.local_gpu import load_examples, run_name, sft_kwargs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/training.jsonl"))
    p.add_argument("--out", type=Path, default=Path("models/agent_qwen3_4b"))
    p.add_argument("--max-steps", type=int, default=5, help="0 trains 2 epochs")
    p.add_argument("--method", choices=["lora", "full"], default="lora")
    p.add_argument("--min-examples", type=int, default=20)
    p.add_argument("--base-model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-length", type=int, default=4096)
    args = p.parse_args()

    config.load_dotenv()
    cfg = config.load()
    examples = load_examples(args.data, args.min_examples)

    import mlflow
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this script needs a local NVIDIA GPU")
    device = torch.cuda.get_device_name(0)
    mlflow.set_tracking_uri(f"databricks://{cfg.profile}")
    mlflow.set_experiment(cfg.experiment)

    split = Dataset.from_list(examples).train_test_split(test_size=0.1, seed=7)
    print(f"{len(split['train'])} train / {len(split['test'])} eval examples on {device}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda"
    )
    peft_config = None
    if args.method == "lora":
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    sft = SFTConfig(**sft_kwargs(args.method, args.max_steps, "/tmp/sft-local", args.max_length))

    with mlflow.start_run(run_name=run_name(args.method, args.max_steps)) as run:
        mlflow.log_params(
            {
                "training_table": cfg.table("training_input"),
                "examples": len(examples),
                "base_model": args.base_model,
                "method": args.method,
                "device": device,
            }
        )
        trainer = SFTTrainer(
            model=model,
            args=sft,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            processing_class=tok,
            peft_config=peft_config,
        )
        trainer.train()
        final = trainer.model.merge_and_unload() if args.method == "lora" else trainer.model
        args.out.mkdir(parents=True, exist_ok=True)
        final.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        mlflow.log_param("out", str(args.out.resolve()))
        print(f"saved merged weights to {args.out}/ ; MLflow run {run.info.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
