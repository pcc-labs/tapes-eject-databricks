# tapes-eject-databricks

Load your coding-agent sessions into Databricks, ask questions of them in SQL, and optionally fine-tune a model on them.

Sessions captured by [Paper](https://papercompute.com) carry labels: `pushback` where an engineer corrected the agent, `apology` where the agent backtracked, `golden` for sessions worth learning from, `regression` for known failures. This repo syncs sessions and labels into Unity Catalog tables, builds views that answer "which model gets corrected most, in which project, and how is that changing," and uses the same labels to pick training data and eval cases for a fine-tune scored with MLflow.

## How it works

```
Paper sessions + labels
  └─ tapes-eject export   sessions, turns, labels -> data/
  └─ tapes-eject sync     -> Unity Catalog tables + views, MLflow eval dataset
  └─ tapes-eject report   which model / project / week carries each label
  └─ sft_train job        fine-tune Qwen3-4B on serverless GPU (or a local GPU)
  └─ eval_models job      base vs tuned, scored in MLflow
```

Labels decide the data:

- **Training examples:** sessions that produced an outcome and carry no correction or failure label (`pushback`, `apology`, `missing-knowledge`, `model-error`, `observation`, `regression`). Four in five `golden` sessions are training examples too.
- **Eval cases:** every turn an engineer corrected, every `regression` session, and the remaining one in five `golden` sessions. A session is never in both sets.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A Paper account with `paperctl` installed and logged in
- A Databricks workspace with Unity Catalog, a SQL warehouse, and a catalog you can create a schema in
- The [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html)
- For fine-tuning: serverless GPU (AI Runtime) in the workspace, or a local NVIDIA GPU (tested on a 32 GB RTX 5090). Databricks Free Edition has no GPUs; a free trial workspace does.

## Setup

```bash
git clone https://github.com/pcc-labs/tapes-eject-databricks
cd tapes-eject-databricks
uv sync

paperctl login
paperctl status                    # expect "auth: healthy"

databricks auth login --host https://<workspace>.cloud.databricks.com --profile tapes-eject
databricks warehouses list --profile tapes-eject      # copy a warehouse id
databricks catalogs list --profile tapes-eject        # pick a catalog

cp .env.example .env               # set TAPES_EJECT_CATALOG and DATABRICKS_WAREHOUSE_ID
uv run tapes-eject doctor
```

`doctor` checks paperd, your Paper org, Databricks auth, the warehouse, the catalog, and MLflow. Every line should say `ok`.

## Usage

### 1. Label sessions

Add labels in the Paper console: mark a few good sessions `golden`, and failures `regression`. Corrections (`pushback`) are what the reports and evals are built around.

If you run an autolabel service, `label` finds a label across recent sessions and can write it back to Paper:

```bash
uv run tapes-eject label pushback --sessions 500            # report matches only
uv run tapes-eject label pushback --sessions 500 --apply    # write them to Paper
```

Set `AUTOLABEL_URL` in `.env` to point at it.

### 2. Export and sync

```bash
uv run tapes-eject export     # sessions, turns, and labels into data/
uv run tapes-eject count      # how many training examples and eval cases the labels select
uv run tapes-eject sync       # tables, views, and the MLflow eval dataset
```

`export` caches every session in `data/cache/`, so a re-run only fetches sessions that changed. It skips sessions over the size caps in `.env`. Text is redacted for common secret shapes before it is written. Redaction is pattern-based, so check `data/` before syncing it anywhere shared.

`sync` creates schema `agent_sessions` in your catalog with:

| Object | Contents |
|---|---|
| `sessions`, `turns`, `labels` | The exported rows. Labels removed in Paper are deleted on the next sync. |
| `training_input` | Training examples, one chat per row |
| `labels_by_model`, `labels_by_project` | Each label's rate over every exported session for that model or project |
| `labels_by_week` | Labeled sessions per week |
| `corrections` | Each corrected turn with the engineer's words, model, and project |
| `eval_cases` (MLflow dataset) | Eval cases with grading guidelines |

`sync` refuses an export where some sessions failed, or one with no eval cases, because syncing it would delete good rows. Re-run `export`, or pass `--force`.

### 3. Ask questions

```bash
uv run tapes-eject report                               # each label's rate by model
uv run tapes-eject report --by project --label pushback
uv run tapes-eject report --by week
```

The same views work in the SQL editor or an AI/BI dashboard. `SELECT * FROM <catalog>.agent_sessions.corrections` shows the moments engineers stepped in.

### 4. Fine-tune and evaluate on Databricks

```bash
databricks bundle deploy                                    # uploads train/, creates the two jobs
databricks bundle run sft_train --params max_steps=5       # smoke run
databricks bundle run sft_train --params max_steps=0       # full run: 2 epochs
databricks bundle run eval_models --params limit=5         # smoke eval on 5 cases
databricks bundle run eval_models --params limit=0         # every case
```

`sft_train` fine-tunes Qwen3-4B on one H100, logs to MLflow experiment `/Shared/tapes-eject`, and registers `<catalog>.agent_sessions.agent_qwen3_4b`. `eval_models` scores base and tuned on `eval_cases` with an MLflow guidelines judge. In MLflow, select the `eval-base` and `eval-tuned` runs and choose **Compare**.

If your workspace only offers A10 GPUs, change `GPU_1xH100` to `GPU_1xA10` in `resources/jobs.yml` and pass `method=lora` to `sft_train`. The catalog defaults to `workspace`; override it with `databricks bundle deploy --var catalog=<catalog>`.

To serve the tuned model:

```bash
uv run tapes-eject serve --version <model version>     # A10 endpoint that scales to zero
uv run tapes-eject ask "Add a --dry-run flag to the export command"
uv run tapes-eject unserve                              # delete it when done
```

### 4b. Fine-tune on a local GPU instead

The data, eval dataset, and experiment stay in Databricks; only the compute moves.

```bash
uv sync --group local                                           # torch (CUDA 12.8), trl, peft
uv run --group local python train/local_sft.py --max-steps 5   # smoke run, LoRA
uv run --group local python train/local_sft.py --max-steps 0   # 2 epochs
uv run --group local python train/local_eval.py --limit 5      # base vs tuned
uv run --group local python train/local_serve.py               # http://127.0.0.1:8081
uv run tapes-eject ask --url http://127.0.0.1:8081 "Add a --dry-run flag to the export command"
```

Tuned weights are saved to `models/agent_qwen3_4b/` and are not registered in Unity Catalog. A full fine-tune of a 4B model needs about 48 GB, so LoRA is the default.

## Commands

| Command | What it does |
|---|---|
| `doctor` | Checks paperd, the Paper org, and the Databricks workspace |
| `label <name> [--apply]` | Finds a label through an autolabel service; `--apply` writes it to Paper |
| `export [--allow-partial]` | Pulls sessions and labels from Paper into `data/` |
| `count` | Shows how much training and eval data the labels select |
| `sync [--force]` | Loads `data/` into Unity Catalog and the MLflow eval dataset |
| `report [--by model\|project\|week] [--label L]` | Prints a label view |
| `serve --version N` / `unserve` / `ask "<prompt>"` | Creates, deletes, and queries the model serving endpoint |
| `spend [--since YYYY-MM-DD]` | Databricks spend from the system billing tables |

## Configuration

All settings live in `.env`; see `.env.example`.

| Variable | Purpose |
|---|---|
| `TAPES_EJECT_CATALOG` | Catalog to create `agent_sessions` in (required) |
| `DATABRICKS_WAREHOUSE_ID` | SQL warehouse that runs the loads (required) |
| `DATABRICKS_CONFIG_PROFILE` | CLI profile, default `tapes-eject` |
| `PAPER_ORG_SLUG` | Paper org to read; default is paperctl's active org |
| `TAPES_EJECT_SAMPLE_SESSIONS` | Unlabeled recent sessions to export alongside labeled ones. They are the denominator for the per-model rates. |
| `TAPES_EJECT_MAX_TURNS`, `TAPES_EJECT_MAX_OUTPUT_TOKENS` | Skip sessions larger than this |
| `TAPES_EJECT_SKIP_SESSIONS` | Comma-separated session ids never to export |

## Troubleshooting

- **`bundle run` fails with `RESOURCE_EXHAUSTED ... GPU quota ... is 0`.** The workspace has no serverless GPU. Use a trial workspace with GPU access, or the local GPU path.
- **`bundle run` says "unknown resource".** Run `databricks bundle deploy` first.
- **`sync` says "refusing to sync".** The export was partial or produced no eval cases. Read `data/report.json`.
- **`sft_train` fails with "only N training examples".** Label more sessions `golden`, then run `export` and `sync` again.
- **Export fails with "could not reach".** Paper's export is not answering. Wait and re-run; cached sessions are not fetched again. If one very large session is the cause, lower `TAPES_EJECT_MAX_OUTPUT_TOKENS` or add its id to `TAPES_EJECT_SKIP_SESSIONS`.
- **`spend` says the billing tables are unavailable.** Reading `system.billing` needs an account admin grant. Use Account console, Usage.

When you are done, run `uv run tapes-eject unserve` and check `uv run tapes-eject spend`.

## Development

```bash
uv run pytest
uv run ruff check .
```

## License

MIT
