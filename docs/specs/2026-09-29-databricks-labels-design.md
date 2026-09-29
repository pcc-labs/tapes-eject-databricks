# Paper labels → Databricks datasets, training, and evals

Date: 2026-09-29
Status: draft for review

## Positioning

**Labeling is Paper.** Paper captures agent sessions across harnesses (Claude Code, Codex, Cursor, Pi) and labels them, both automatically with detectors and by hand. Datasets, training, and evals are not Paper's product. That is where Databricks shines.

The demo's one line: **Paper labels your agent work. Databricks turns the labels into datasets, evals, and models.**

Databricks has no answer for capturing and judging coding-agent work. Paper fills that gap, and everything downstream of the label runs on Databricks at full strength: Unity Catalog governance, MLflow evaluation, GPU training, Model Serving.

Only two products appear in the demo: Paper and Databricks. The fine-tuning code is an implementation detail. On stage it is "a supervised fine-tune (SFT) of an open-weight model on Databricks" and it is never presented as a product.

## Audience

A developer who is new to Databricks. They know agents and Python. They do not know Unity Catalog, MLflow, jobs, or Model Serving.

That shapes every choice:

- Everything runs from the terminal. The Databricks UI is where you look at results, not where you do the work.
- Each act introduces one Databricks concept, defined in one line the first time it appears.
- No notebooks are required to follow along.

## The story: four acts

### Act 1: Capture and label (Paper)

Real sessions sit in the console. The Auto label button (paperplane#51, autolabel-cassette) applies `pushback`, `apology`, `no-outcome`, and the other detectors. A person marks a handful of sessions `golden` (behavior we want to keep) or `regression` (a failure that must not come back).

Takeaway: labeling is Paper's job. Every downstream act depends on these labels.

### Act 2: Curate (Paper → Unity Catalog)

`sync` copies labeled sessions into Unity Catalog ("tables with permissions, history, and lineage"). It writes two things:

- Delta tables of sessions and labels, plus a `training_input` view of the sessions the labels select for training.
- An MLflow evaluation dataset ("a versioned set of test cases, each with its inputs and expected result") built from `golden` and `regression` sessions.

Show the dataset's version history and lineage in the UI.

Takeaway: the label in Paper is the one place a person decides what counts as good. Databricks keeps a controlled, versioned record of that decision.

### Act 3: Train (Databricks)

A Databricks job ("a run on Databricks compute, started on demand or on a schedule") on AI Runtime ("serverless GPUs, billed only while the job runs") post-trains Qwen3-4B with supervised fine-tuning on the curated view:

1. Read `training_input`, which is already in chat-messages format.
2. Train with TRL's `SFTTrainer` on one H100. This follows Databricks' own Qwen3-4B SFT tutorial, so the demo shows their recommended approach.
3. Checkpoints go to a Unity Catalog volume ("a governed folder for files"). Metrics log to MLflow automatically.
4. Register the model in Unity Catalog.

Only SFT is used, with no DPO. The labels decide which sessions count as good examples. SFT on those examples is the whole training step.

Takeaway: curated sessions become a model without leaving the platform. Lineage runs from label, to table, to training run, to model.

### Act 4: Prove it (MLflow evaluation)

Serve the base and fine-tuned models on Model Serving ("your model as an HTTPS endpoint"). Run `mlflow.genai.evaluate` over the evaluation dataset for both and compare them side by side in the MLflow UI.

Optional close: point the served endpoint through tapes so the model's own sessions land back in Paper, get labeled, and feed the next cycle.

Takeaway: labels drive both the training data and the proof that training worked.

## Architecture

```
Paper (console + core)              Databricks workspace
─────────────────────               ──────────────────────────────────────────
sessions ─┐                         UC catalog: paper_demo
labels  ──┼─► 01_export ─► JSONL ─► 02_sync ─► schema agent_sessions
          │    (local)               (local,    ├─ sessions        (Delta)
          │                          SDK)       ├─ turns           (Delta)
          │                                     ├─ labels          (Delta)
          │                                     ├─ training_input  (view)
          │                                     └─ eval_cases      (MLflow eval dataset)
          │                                              │
          │                         03_train (job, GPU) ◄┘ training_input
          │                            └─► MLflow run ─► UC model: paper_demo.agent_sessions.agent_qwen3_4b
          │                                              │
          │                         04_eval (local) ◄────┘ Model Serving endpoints (base, tuned)
          │                            └─► mlflow.genai.evaluate over eval_cases
          └──────────── optional: tuned endpoint proxied through tapes ◄───────┘
```

## Components

All live in this repo, `tapes-eject-databricks`.

| Path | What it does |
|---|---|
| `01_export.py` | Pulls sessions (full-fidelity export, same shape tapes stores) and their labels from Paper into `data/sessions.jsonl` and `data/labels.jsonl`. |
| `02_sync.py` | Loads both into the UC tables with `MERGE`, then rebuilds `training_input` and syncs `eval_cases`. |
| `03_train/` | A Databricks Asset Bundle with the job definition and the SFT script (adapted from Databricks' Qwen3-4B tutorial), started with `databricks bundle run` from the terminal. |
| `04_eval.py` | Serves or uses the endpoints, runs `mlflow.genai.evaluate` for base and tuned, and prints the comparison and the UI link. |
| `README.md` | Setup and each act, written for someone new to Databricks. |
| `RUNBOOK.md` | The live demo script: what to click, what to say, and what to have pre-run. |

## Data model

- **`sessions`**: one row per session. Columns: `session_id`, `harness`, `model`, `started_at`, `total_cost`, `outcome`, plus a `raw` column holding the export record.
- **`turns`**: one row per turn. Columns: `session_id`, `turn_id`, `ordinal`, `user_text`, `agent_text`, `raw`.
- **`labels`**: one row per label attachment. Columns: `session_id`, `turn_id` (nullable), `span_id` (nullable), `label`, `source` (`human` or `auto`), `synced_at`.
- **`training_input`**: a view with one row per training example in chat-messages format (`messages: [{role, content}]`), the format `SFTTrainer` reads. Labels alone decide inclusion: a session is in if it is labeled `golden`, or it has an outcome and carries no `pushback`, `apology`, or `regression` label. Sessions labeled `regression` stay out of training and only feed `eval_cases`.
- **`eval_cases`**: one record per `golden` or `regression` session.
  - `inputs`: the opening user request plus context.
  - `expectations`, golden: the observed good outcome.
  - `expectations`, regression: the failure to avoid, from the session's labels.

## Sync semantics

- Keyed on `(session_id, turn_id, span_id, label)`. Rerunning `sync` is idempotent.
- A label removed in Paper is deleted from `labels` on the next sync. `eval_cases` is rebuilt from the current labels, and MLflow keeps the prior version.
- `sync` never writes back to Paper. Paper stays the source of truth for labels.

## Scoring in Act 4

- Golden cases: an LLM judge checks each answer against the expected outcome, using MLflow's built-in scorers.
- Regression cases: a guideline scorer asks "does the answer repeat the failure described?", and the case passes when it does not.
- The judge model comes from Databricks Foundation Model APIs, so no outside key is needed.

## Risks and decisions

1. **GPU access.** Act 3 runs on AI Runtime, which has been in public preview since 2026-03-19. The workspace needs that preview turned on and H100s available in its region. Full fine-tuning a 4B model needs the 80 GB H100. If only A10s are available, switch to LoRA, following Databricks' LoRA tutorial. Last-resort fallback: train off-platform, but still log to Databricks MLflow and register the model in Unity Catalog. Acts 2 and 4 are unchanged either way.
2. **Serving a fine-tuned model.** Serving a custom LLM on Model Serving needs GPU serving in the workspace. The plan confirms this before Act 4 depends on it.
3. **Data volume.** The count of `golden` and `regression` sessions sets how convincing Act 4 is. The runbook states the counts on screen rather than hiding them. The plan's first task counts what exists.
4. **Labels are free-form.** `golden` and `regression` are ordinary labels a person creates in the console. The Labels feature already supports that. No product change is needed.
5. **API surface.** Export and label reads use Paper's existing endpoints (session export, and core's `?label=` filter). The plan confirms the exact calls before writing `01_export`.
6. **Omnigent** is out of scope. It can be added once harness support lands.

## Out of scope

- Any change to the console or paperplane.
- DPO or any other preference training.
- The old Foundation Model Fine-tuning API, which is end-of-life.
- Writing labels from Databricks back to Paper.
- Training a replacement for the TypeSafe judge.
- Omnigent capture.

## Success criteria

- A developer new to Databricks can run Acts 2–4 from the README against their own workspace.
- The live demo runs in 15 minutes or less. Training is pre-run, and Act 3 shows the finished run and its lineage.
- Every number shown comes from a real run, and the runbook says where each one comes from.
- Removing a label in Paper and rerunning `sync` visibly changes the next evaluation-dataset version.
