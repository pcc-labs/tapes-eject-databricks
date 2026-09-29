# tapes-eject-databricks

Paper labels your agent sessions. This demo takes those labels to Databricks, where they become datasets, evals, and a fine-tuned model.

- Act 1: capture and label sessions in Paper
- Act 2: sync the labeled sessions into Unity Catalog and an MLflow evaluation dataset
- Act 3: supervised fine-tune (SFT) of Qwen3-4B on Databricks AI Runtime GPUs, using the sessions the labels selected
- Act 4: evaluate the base and fine-tuned models against `golden` and `regression` cases in MLflow

Design: [docs/specs/2026-09-29-databricks-labels-design.md](docs/specs/2026-09-29-databricks-labels-design.md)

Status: design only. No code yet.

## Setup

You need: a Databricks free-trial workspace (not Free Edition, which has no GPUs), Paper access, and the autolabel cassette running beside this repo.

1. Run every command in "Step 1" of `docs/superpowers/plans/2026-09-29-tapes-eject-databricks.md`.
2. `cp .env.example .env` and fill in `TAPES_EJECT_CATALOG` and `DATABRICKS_WAREHOUSE_ID`.
3. `uv sync`
4. `uv run tapes-eject doctor`. Every line should say `ok`.

A few Databricks words, defined once:
- **Unity Catalog**: where tables, files, and models live, with permissions and history. Names are `catalog.schema.thing`.
- **Volume**: a governed folder of files inside Unity Catalog, at `/Volumes/<catalog>/<schema>/<volume>`.
- **SQL warehouse**: compute that runs SQL. We use it to load tables.
- **MLflow**: the experiment tracker. It records training runs, evaluation datasets, and scores.
- **AI Runtime**: serverless GPUs. You pay only while a job runs.
- **Job**: a run of a notebook or script on Databricks compute, started from the CLI.
- **Model Serving**: your model as an HTTPS endpoint.

## Act 3: Train on Databricks GPUs

`databricks bundle run sft_train --params max_steps=5` is the smoke run; `max_steps=0` is the full run. The job reads the training examples the labels selected, fine-tunes Qwen3-4B with supervised fine-tuning on one H100, logs the run to MLflow, and registers `<catalog>.agent_sessions.agent_qwen3_4b` in Unity Catalog.
