# Databricks notebook source
# MAGIC %md
# MAGIC # SFT: Qwen3-4B on the sessions Paper's labels selected
# MAGIC Runs on AI Runtime (serverless GPU, 1x H100, AI environment v6+). Adapted from Databricks'
# MAGIC "Full fine-tuning of Qwen3-4B" tutorial. `max_steps=5` is the smoke run; `0` trains 2 epochs.

# COMMAND ----------
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "agent_sessions")
dbutils.widgets.text("max_steps", "5")
dbutils.widgets.text("method", "full")  # full on H100, lora on A10
dbutils.widgets.text("min_examples", "20")
dbutils.widgets.text("base_model", "Qwen/Qwen3-4B")
dbutils.widgets.text("experiment", "/Shared/tapes-eject")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MAX_STEPS = int(dbutils.widgets.get("max_steps"))
METHOD = dbutils.widgets.get("method")
MIN_EXAMPLES = int(dbutils.widgets.get("min_examples"))
BASE = dbutils.widgets.get("base_model")
assert CATALOG, "set the `catalog` job parameter"
assert METHOD in ("full", "lora"), METHOD

VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
OUT = f"{VOLUME}/models/agent_qwen3_4b"
UC_MODEL = f"{CATALOG}.{SCHEMA}.agent_qwen3_4b"

# COMMAND ----------
import json

with open(f"{VOLUME}/training.jsonl", encoding="utf-8") as fh:
    examples = [{"messages": json.loads(line)["messages"]} for line in fh if line.strip()]
if len(examples) < MIN_EXAMPLES:
    raise ValueError(
        f"only {len(examples)} training examples (need {MIN_EXAMPLES}); label more sessions "
        "`golden` in Paper, then `tapes-eject export && tapes-eject sync`"
    )

from datasets import Dataset

split = Dataset.from_list(examples).train_test_split(test_size=0.1, seed=7)
print(f"{len(split['train'])} train / {len(split['test'])} eval examples")

# COMMAND ----------
import importlib.util

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("experiment"))

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
peft_config = None
if METHOD == "lora":
    from peft import LoraConfig

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

args = SFTConfig(
    output_dir="/tmp/sft",
    max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=1e-5 if METHOD == "full" else 1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    max_length=4096,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="no",
    report_to="mlflow",
    use_liger_kernel=importlib.util.find_spec("liger_kernel") is not None,
)

run_name = f"sft-{METHOD}-{MAX_STEPS or 'full'}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({"training_table": f"{CATALOG}.{SCHEMA}.training_input", "examples": len(examples), "base_model": BASE, "method": METHOD})
    trainer = SFTTrainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"], processing_class=tok, peft_config=peft_config)
    trainer.train()
    final = trainer.model.merge_and_unload() if METHOD == "lora" else trainer.model
    final.save_pretrained(OUT)
    tok.save_pretrained(OUT)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register for Model Serving
# MAGIC Logged from this GPU job on purpose: logging from CPU packages CPU dependencies and the GPU
# MAGIC serving endpoint fails to start. Pattern from Databricks' "Serve custom LLMs" docs.

# COMMAND ----------
import shutil

from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel

shutil.copytree(OUT, "agent_model", dirs_exist_ok=True)


class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


metadata = {
    "task": "llm/v1/chat",
    "entrypoint": (
        "python -u -m vllm.entrypoints.openai.api_server "
        "--model agent_model --served-model-name agent "
        "--host 0.0.0.0 --port 8080 "
        "--dtype float16 --max-model-len 8192 "
        "--gpu-memory-utilization 0.85"
    ),
}

with mlflow.start_run(run_name=f"{run_name}-register"):
    info = mlflow.pyfunc.log_model(
        name="agent_qwen3_4b",
        python_model=LLMModel(),
        artifacts={"model_dir": "agent_model"},
        metadata=metadata,
        extra_pip_requirements=["mlflow==3.12.0"],
    )
    version = mlflow.register_model(info.model_uri, UC_MODEL, env_pack="databricks_model_serving")
print(f"registered {UC_MODEL} version {version.version}")
dbutils.notebook.exit(json.dumps({"model": UC_MODEL, "version": version.version, "out": OUT}))
