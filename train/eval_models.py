# Databricks notebook source
# MAGIC %md
# MAGIC # Base vs tuned on the eval cases Paper's labels built
# MAGIC Runs on AI Runtime (serverless GPU, 1x A10 is enough for a 4B model). Each case is judged
# MAGIC against its own guideline by MLflow's ExpectationsGuidelines scorer.

# COMMAND ----------
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "agent_sessions")
dbutils.widgets.text("base_model", "Qwen/Qwen3-4B")
dbutils.widgets.text("experiment", "/Shared/tapes-eject")
dbutils.widgets.text("limit", "0")  # 0 = every case

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
assert CATALOG, "set the `catalog` job parameter"
TUNED = f"/Volumes/{CATALOG}/{SCHEMA}/raw/models/agent_qwen3_4b"
LIMIT = int(dbutils.widgets.get("limit"))

# COMMAND ----------
import os

# One generation at a time: mlflow.genai.evaluate calls predict_fn from a thread pool (10 by
# default), and ten concurrent generate() calls on one GPU run out of memory.
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"

import mlflow
import torch
from mlflow.genai.scorers import ExpectationsGuidelines
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_INPUT_TOKENS = 8192

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("experiment"))
dataset = mlflow.genai.datasets.get_dataset(name=f"{CATALOG}.{SCHEMA}.eval_cases")
# The dataset object keeps eval-run -> dataset lineage; a sampled smoke run passes a frame.
cases = dataset.to_df().head(LIMIT) if LIMIT else dataset
print(f"{LIMIT or len(dataset.to_df())} eval cases")


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")

    def predict(messages):
        enc = tok.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        # Keep the most recent context when a conversation is longer than the cap.
        enc = {k: v[:, -MAX_INPUT_TOKENS:].to("cuda") for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=512, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[-1] :], skip_special_tokens=True)

    return predict, model


# COMMAND ----------
for name, path in (("base", dbutils.widgets.get("base_model")), ("tuned", TUNED)):
    predict, model = load(path)
    with mlflow.start_run(run_name=f"eval-{name}"):
        result = mlflow.genai.evaluate(
            data=cases, predict_fn=predict, scorers=[ExpectationsGuidelines()]
        )
        print(name, result.metrics)
    del model
    torch.cuda.empty_cache()
