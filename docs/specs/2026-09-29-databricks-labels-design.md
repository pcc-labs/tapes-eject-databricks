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

Real sessions sit in the console, already labeled in paperd. Live on stage, the autolabel cassette finds a label across a page of sessions, then applies it: `POST /run` with `apply: false`, show the count, then the same call with `apply: true`. The labels land on the console's own rows. A person marks a handful of sessions `golden` (behavior we want to keep) or `regression` (a failure that must not come back).

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

## Where the data comes from

**paperd is the source of truth.** Every read goes through the local paperd proxy, which supplies auth and routing. Nothing reads the autolabel-cassette repo's `labels/*.jsonl` files, which are that tool's working copy, not the record.

| Need | Surface |
|---|---|
| Label names and counts | labels cassette: `list-labels`, `label-usage` |
| Which sessions, traces (turns), and spans carry a label | labels cassette: `list-label-attachments`, per `primitive_type` (`session`, `trace`, `span`), paged by cursor |
| The full session record | export cassette / `GET /v1/sessions/{id}/traces`, the same record `paperctl sessions export` writes |
| Labeling live, and per-turn evidence | autolabel cassette: `POST /v1/cassettes/autolabel/run {label, session_ids, apply}` answers `202 {id}`, then poll `GET /v1/cassettes/autolabel/runs/{id}` for `{state, progress, result}`. The result gives each session's matched turns with their `evidence`. |

**The autolabel cassette is not in the org's deployment yet.** The TKO grant (papercomputeco/cloud#271) was closed, so paperd does not front it. For the demo it runs at its own URL: local `python -m label_sampler serve` on `:9996`, or the EC2 box from autolabel-cassette#4. `TYPESAFE_API_KEY` must be set on it, or `pushback`, `question`, and `observation` answer `needs_judge`. `01_export` takes the cassette's base URL as configuration.

**What paperd holds today** (org `papercomputeco`, read with `paperctl label list` on 2026-09-29):

| Label | Sessions | Turns (traces) | Spans |
|---|---|---|---|
| `pushback` | 26 | 42 | 42 |
| `question` | 21 | 54 | 54 |
| `observation` | 15 | 29 | 29 |
| `apology` | 9 | 12 | 12 |
| `missing-knowledge` | 10 | 25 | 77 |
| `model-error` | 1 | 2 | 16 |
| `subagents` | 34 | 45 | 45 |
| `no-outcome` | 5 | | |

There are also human labels such as `design exploration` and `🦾 Potential Skill`. `golden` and `regression` do not exist yet. A person creates them in the console during Act 1.

## Components

All live in this repo, `tapes-eject-databricks`.

| Path | What it does |
|---|---|
| `01_export.py` | Through paperd: lists labels and their attachments at session, trace, and span level, then pulls the full record of every labeled session plus an unlabeled sample. Optionally asks the autolabel cassette (`apply: false`) for per-turn evidence. Writes `data/sessions.jsonl` and `data/labels.jsonl`. |
| `02_sync.py` | Loads both into the UC tables with `MERGE`, then rebuilds `training_input` and syncs `eval_cases`. |
| `03_train/` | A Databricks Asset Bundle with the job definition and the SFT script (adapted from Databricks' Qwen3-4B tutorial), started with `databricks bundle run` from the terminal. |
| `04_eval.py` | Serves or uses the endpoints, runs `mlflow.genai.evaluate` for base and tuned, and prints the comparison and the UI link. |
| `README.md` | Setup and each act, written for someone new to Databricks. |
| `RUNBOOK.md` | The live demo script: what to click, what to say, and what to have pre-run. |

## Data model

- **`sessions`**: one row per session. Columns: `session_id`, `harness`, `model`, `started_at`, `total_cost`, `outcome`, plus a `raw` column holding the export record.
- **`turns`**: one row per turn. Columns: `session_id`, `turn_id`, `ordinal`, `user_text`, `agent_text`, `raw`.
- **`labels`**: one row per label attachment, as paperd stores it. Columns: `label`, `primitive_type` (`session`, `trace`, or `span`), `primitive_id`, `session_id`, `turn_id` (the trace id, nullable), `span_id` (nullable), `evidence` (from the autolabel cassette, nullable), `synced_at`.
- **`training_input`**: a view with one row per training example in chat-messages format (`messages: [{role, content}]`), the format `SFTTrainer` reads. Labels alone decide inclusion: a session is in if it is labeled `golden`, or it has an outcome and carries none of `pushback`, `apology`, `missing-knowledge`, `model-error`, or `regression`. Excluded sessions stay out of training and feed `eval_cases`.
- **`eval_cases`**: built mostly from labels that already exist, so Act 4 does not wait on new labeling.
  - **Correction cases.** Each turn labeled `pushback`, `observation`, or `missing-knowledge` becomes a case. `inputs`: the conversation up to the agent turn the engineer corrected. `expectations`: the engineer's correction, which is the fact or direction the agent should have followed without being told. Today that is about 96 turns before de-duplication (42 + 29 + 25).
  - **`golden` sessions.** `inputs`: the opening request. `expectations`: the observed good outcome.
  - **`regression` sessions.** `inputs`: the opening request. `expectations`: the failure to avoid.

## Sync semantics

- Keyed on `(session_id, turn_id, span_id, label)`. Rerunning `sync` is idempotent.
- A label removed in Paper is deleted from `labels` on the next sync. `eval_cases` is rebuilt from the current labels, and MLflow keeps the prior version.
- `sync` never writes back to Paper. Paper stays the source of truth for labels.

## Scoring in Act 4

- Golden cases: an LLM judge checks each answer against the expected outcome, using MLflow's built-in scorers.
- Regression cases: a guideline scorer asks "does the answer repeat the failure described?", and the case passes when it does not.
- The judge model comes from Databricks Foundation Model APIs, so no outside key is needed.

## Risks and decisions

1. **GPU access.** Act 3 runs on AI Runtime, which has been in public preview since 2026-03-19, in a Databricks free-trial workspace ($400 credits). Free Edition is ruled out because it has no GPUs and no GPU serving. The plan's first task confirms that the trial workspace can start an AI Runtime H100 job. Full fine-tuning a 4B model needs the 80 GB H100. If only A10s are available, switch to LoRA, following Databricks' LoRA tutorial. Last-resort fallback: LoRA SFT on a local RTX 5090, still logging to Databricks MLflow and registering in Unity Catalog. The training script is written so the same code runs in both places.
2. **Serving a fine-tuned model.** Act 4 needs a GPU Model Serving endpoint for the tuned model. The plan confirms the trial workspace allows one before Act 4 depends on it. If it does not, serve on the 5090 with vLLM and log the evaluation to Databricks MLflow.
3. **Data volume.** About 96 correction turns across roughly 50 sessions exist today, which is enough for a credible evaluation set. The SFT set is the open question: it depends on how many sessions have an outcome and no negative label. The plan's first task counts both from paperd. The runbook shows the counts on screen rather than hiding them.
4. **Labels are free-form.** `golden` and `regression` are ordinary labels a person creates in the console. The Labels feature already supports that. No product change is needed.
5. **Reading through paperd.** Reads use paperd's labels and export cassettes, which paperd authenticates. The autolabel cassette runs outside the deployment (see "Where the data comes from"). The plan's first task confirms both from a script, not only the CLI.
6. **Omnigent** is out of scope. It can be added once harness support lands.

## Budget

Everything fits in the trial's $400. We are showing Databricks at full strength, so the demo uses their GPUs rather than working around them. Guardrails:

- Training uses a capped `max_steps` and a job timeout. Every run is sized from a short first measurement before any full run.
- Serving endpoints scale to zero, run only for the Act 4 evaluation and the live demo, and are deleted afterwards.
- The Act 4 judge uses pay-per-token Foundation Model APIs over a small evaluation set.
- The plan records actual spend after each act, from Databricks' billing usage table, so we know the headroom before the live demo.

## Out of scope

- Any change to the console or paperplane.
- Admitting the autolabel cassette into the org's deployment.
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
