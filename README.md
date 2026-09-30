# tapes-eject-databricks

Paper labels your agent sessions. This demo takes those labels to Databricks, where they become datasets, evals, and a fine-tuned model.

- **Act 1:** capture and label sessions in Paper.
- **Act 2:** sync the labeled sessions into Unity Catalog and an MLflow evaluation dataset.
- **Act 3:** run a supervised fine-tune (SFT) of Qwen3-4B on Databricks AI Runtime GPUs, using the sessions the labels selected.
- **Act 4:** score the base and fine-tuned models against the eval cases in MLflow.

Nothing here is a new Paper feature. The demo builds on Paper's session export, Labels, and the autolabel cassette. "Eject" is only the demo's name.

**Status:** code complete, with 59 unit tests passing. It has not yet run end to end against a Databricks workspace. `RUNBOOK.md` is the live demo script. The design is in [docs/specs/2026-09-29-databricks-labels-design.md](docs/specs/2026-09-29-databricks-labels-design.md), and the build plan is in [docs/superpowers/plans/2026-09-29-tapes-eject-databricks.md](docs/superpowers/plans/2026-09-29-tapes-eject-databricks.md).

## A few Databricks words

- **Unity Catalog**: where tables, files, and models live, with permissions and history. Names are `catalog.schema.thing`.
- **Volume**: a governed folder of files inside Unity Catalog, at `/Volumes/<catalog>/<schema>/<volume>`.
- **SQL warehouse**: compute that runs SQL. This demo uses it to load tables.
- **MLflow**: the experiment tracker. It records training runs, evaluation datasets, and scores.
- **AI Runtime**: serverless GPUs. You pay only while a job runs.
- **Job**: a run of a notebook or script on Databricks compute, started from the CLI.
- **Model Serving**: your model as an HTTPS endpoint.

## What you need

- **A Databricks free-trial workspace, which comes with $400 of credits.** Free Edition won't work because it has no GPUs.
- **Paper access**, with `paperctl` logged in to the org whose sessions you want to use.
- **A TypeSafe API key.** The autolabel cassette uses it to judge `pushback`, `question`, and `observation`. Without it, those three labels answer `needs_judge`.
- **These tools:** `uv`, `gh`, and git.

## Setup (once per machine)

### 1. Tools and logins

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# GitHub: the autolabel-cassette dependency is a private pcc-labs repo
gh auth login
gh auth setup-git

# Paper
paperctl login
paperctl init
paperctl status      # expect "auth: healthy"
paperctl whoami      # note org_slug (papercomputeco is zro54)

# Databricks CLI, logged in to your trial workspace
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile tapes-eject
databricks warehouses list --profile tapes-eject   # note the SQL warehouse id
databricks catalogs list --profile tapes-eject     # pick a catalog you can create a schema in
```

### 2. The autolabel cassette, beside this repo

Keep it running in its own terminal for the whole demo.

```bash
git clone https://github.com/pcc-labs/autolabel-cassette ../autolabel-cassette
cd ../autolabel-cassette
git checkout 1a43a65b969a40cc41892f451a13850905e919cb   # the revision pinned in pyproject.toml
uv sync
printf 'LABEL_SAMPLER_ORG=zro54\nTYPESAFE_API_KEY=<key>\n' > .env
uv run python -m label_sampler serve   # http://127.0.0.1:9996/v1/cassettes/autolabel
```

Newer cassette revisions serve at `/api/autolabel` instead. Either keep the clone at the pinned revision, or set `AUTOLABEL_URL` to the prefix the cassette prints when it starts.

### 3. This repo

```bash
git clone https://github.com/pcc-labs/tapes-eject-databricks
cd tapes-eject-databricks
cp .env.example .env    # fill in TAPES_EJECT_CATALOG and DATABRICKS_WAREHOUSE_ID
uv sync
uv run tapes-eject doctor
```

`doctor` checks seven things: paperd, your Paper org, the cassette (including that its org matches), Databricks auth, the SQL warehouse, the catalog, and MLflow. Every line should say `ok` before you go on.

### 4. Check that the workspace has GPUs

1. In the workspace UI, go to **New → Notebook**.
2. In the compute selector, pick **Serverless GPU**.
3. In the Environment panel, choose accelerator **1xH100** and environment **AI v6**, then click **Apply**.
4. Run a cell containing `%sh nvidia-smi`. You should see `NVIDIA H100 80GB HBM3`.

If only **A10** is offered, see the note in step 5.

### 5. Deploy the bundle, including the two GPU jobs

```bash
databricks bundle deploy    # uploads train/ and creates the sft_train and eval_models jobs
```

`resources/jobs.yml` defines both jobs on AI Runtime: `sft_train` on one H100 and `eval_models` on one A10, both with the AI v6 base environment and a 2 h timeout. The serverless-GPU settings are the task's `compute.hardware_accelerator` and the environment's `base_environment`; both are Beta on the Jobs API. The catalog comes from the bundle variable `catalog` (default `workspace`); override it with `databricks bundle deploy --var catalog=<catalog>`.

If only **A10** is offered in your workspace, change `GPU_1xH100` to `GPU_1xA10` in `resources/jobs.yml` and pass `method=lora` to the training job in Act 3.

## Running the demo

### Act 1: Capture and label (Paper)

```bash
uv run tapes-eject label apology            # finds the label across the 25 newest sessions; writes nothing
uv run tapes-eject label apology --apply    # labels them in Paper; the chips appear in the console
```

Sessions over `TAPES_EJECT_MAX_TURNS` or `TAPES_EJECT_MAX_OUTPUT_TOKENS` are printed as `skip` and never sent to the cassette, because exporting a very large session can take Paper's export service down.

You can use any of the seven labels: `apology`, `dream`, `subagents`, `no-outcome`, `pushback`, `question`, `observation`. In the console, also mark a few good sessions `golden` and one failure `regression` by hand.

### Act 2: Curate (Paper → Unity Catalog)

```bash
uv run tapes-eject export   # Paper's session export + Labels for every labeled session, into data/
uv run tapes-eject count    # what the labels select: training examples, eval cases, golden, regression
uv run tapes-eject sync     # tables + training examples into Unity Catalog, eval cases into MLflow
```

- **`export`** skips sessions over `TAPES_EJECT_MAX_TURNS` or `TAPES_EJECT_MAX_OUTPUT_TOKENS` and lists them under `skipped` in `data/report.json`. It caches each session in `data/cache/`, so re-running it only fetches sessions that changed. It exits non-zero if any export failed or any outcome is unknown. `data/report.json` says which.
- **`sync`** refuses a partial or empty export, because syncing one would wipe good data. Re-run `export`, or pass `--force` if you mean it.
- **What to open afterwards:** Catalog → your catalog → `agent_sessions`, where `labels` → History shows one version per sync. Then MLflow → Datasets → `eval_cases`.

How the labels decide the data:
- **Training data:** sessions that produced an outcome and carry no `pushback`, `apology`, `missing-knowledge`, `model-error`, `observation`, or `regression` label. Four in five `golden` sessions are training data too.
- **Eval cases:** every turn an engineer corrected (`pushback`, `observation`, `missing-knowledge`), every `regression` session, and the other one in five `golden` sessions. A session is never both training data and an eval case.

### Act 3: Train on Databricks GPUs

```bash
databricks bundle run sft_train --params max_steps=5    # smoke run; check the cost first
uv run tapes-eject spend                                # spend so far, out of $400
databricks bundle run sft_train --params max_steps=0    # full run: 2 epochs
```

The job fine-tunes Qwen3-4B on the training examples, logs the run to MLflow experiment `/Shared/tapes-eject`, and registers `<catalog>.agent_sessions.agent_qwen3_4b` in Unity Catalog. If the full fine-tune runs out of memory, add `method=lora`.

### Act 4: Prove it

```bash
databricks bundle run eval_models --params limit=5    # smoke run on 5 cases
databricks bundle run eval_models --params limit=0    # every case
```

In MLflow, select the `eval-base` and `eval-tuned` runs and choose **Compare**. The guideline pass rate is the Act 4 number.

To show the fine-tuned model live:

```bash
uv run tapes-eject serve --version <model version>    # A10 endpoint; scales to zero; a cold start takes minutes
uv run tapes-eject ask "Add a --dry-run flag to the export command"
uv run tapes-eject unserve                            # delete it when the demo is over
```

## Commands

| Command | What it does |
|---|---|
| `doctor` | Checks paperd, the cassette, and the Databricks workspace |
| `label <name> [--apply]` | Finds a label across recent sessions through the autolabel cassette; `--apply` writes it to Paper |
| `export [--evidence] [--allow-partial]` | Pulls labeled sessions and their labels from Paper into `data/` |
| `count` | Shows how much training and eval data the labels select |
| `sync [--force]` | Loads `data/` into Unity Catalog and the MLflow evaluation dataset |
| `serve --version N` / `unserve` / `ask "<prompt>"` | Creates, deletes, and queries the fine-tuned model's endpoint |
| `spend [--since YYYY-MM-DD]` | Shows Databricks spend from the system billing tables |

Run any of these as `uv run tapes-eject <command>`. The unit tests run with `uv run pytest`.

## Troubleshooting

- **`export` fails with "could not reach the tapes API".** Paper's session export endpoint isn't answering. Check with `paperctl sessions export <id>` directly. `sessions list` can work while export is down. Wait for it to recover, then re-run: cached sessions are not fetched again.
- **`export` took down Paper's export service.** A session of a few turns can still carry hundreds of megabytes of tool output. Lower `TAPES_EJECT_MAX_OUTPUT_TOKENS` in `.env`; the default of 400000 is below the largest session known to export cleanly and above the ones that did not. `paperctl sessions list --json` shows each session's `rollup.usage.output_tokens`.
- **`doctor` says the cassette refused the connection, but it is running.** The cassette clone is at a newer revision that serves `/api/autolabel`. Check out the pinned revision (Setup step 2) or set `AUTOLABEL_URL` to the prefix the cassette printed.
- **`label` or `export` prints `needs_judge`.** The cassette has no `TYPESAFE_API_KEY`. Set it in `../autolabel-cassette/.env` and restart the cassette.
- **`doctor` says the cassette org doesn't match.** Set `LABEL_SAMPLER_ORG` in the cassette's `.env` to your `org_slug`.
- **`sync` says it is "refusing to sync".** The export was partial, or it produced no eval cases. Read `data/report.json`.
- **`bundle run` says "unknown resource".** The bundle has not been deployed to this workspace yet: run `databricks bundle deploy`.
- **`sft_train` fails with "only N training examples".** Mark more good sessions `golden` in Paper, then run `export` and `sync` again.
- **`spend` says the billing tables are unavailable.** Reading `system.billing` needs a grant from an account admin. Use Account console → Usage, or the trial credit banner in the workspace, instead.
- **`bundle run` fails with `RESOURCE_EXHAUSTED ... GPU quota ... is 0 node(s)`.** The workspace is Free Edition, which has no GPUs, no account console, and no GPU serving. There is no in-place upgrade: start the free trial at databricks.com/try-databricks with a business email, log the CLI in to the new workspace, update `.env`, and run `databricks bundle deploy` again.

After the demo, run `uv run tapes-eject unserve` and check `uv run tapes-eject spend`.
