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
import mlflow
import torch
from mlflow.genai.scorers import ExpectationsGuidelines
from transformers import AutoModelForCausalLM, AutoTokenizer

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("experiment"))
cases = mlflow.genai.datasets.get_dataset(name=f"{CATALOG}.{SCHEMA}.eval_cases").to_df()
if LIMIT:
    cases = cases.head(LIMIT)
print(f"{len(cases)} eval cases")


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")

    def predict(messages):
        ids = tok.apply_chat_template(
            list(messages), add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        ).to("cuda")
        out = model.generate(ids, max_new_tokens=512, do_sample=False)
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    return predict, model


# COMMAND ----------
for name, path in (("base", dbutils.widgets.get("base_model")), ("tuned", TUNED)):
    predict, model = load(path)
    with mlflow.start_run(run_name=f"eval-{name}"):
        result = mlflow.genai.evaluate(data=cases, predict_fn=predict, scorers=[ExpectationsGuidelines()])
        print(name, result.metrics)
    del model
    torch.cuda.empty_cache()
