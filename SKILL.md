---
name: tapes-eject-databricks
description: Use when the user wants their Codex or Claude Code sessions in Databricks, wants to know which model or project they correct most, wants to fine-tune on their own sessions with Databricks, or wants to set up, debug, or run tapes-eject-databricks. Covers importing history with tapes, labeling pushback, golden, and regression sessions, syncing to Unity Catalog, label reports, and the sft_train and eval_models jobs.
---

# Agent history in Databricks

`tapes-eject` reads Codex and Claude Code sessions from a local tapes stack,
labels the turns where the person corrected the agent, and syncs sessions,
turns, and labels into Unity Catalog. Views there answer which model,
project, or week carries each label. The same labels pick training data and
eval cases for a Qwen3-4B fine-tune scored in MLflow.

Work from the root of this repo. Run every command with `uv run`.

## Order of work

Do these in order. Each step needs the one before it.

1. **History in tapes.** `tapes-skills-demo check`, then `tapes-skills-demo --ollama`.
2. **Repo ready.** `uv sync`, Databricks CLI login, `.env`, `uv run tapes-eject doctor`.
3. **Labels.** `uv run tapes-eject label`, then `mark` by hand.
4. **Data.** `uv run tapes-eject export`, `count`, `sync`, `report`.
5. **Fine-tune (optional).** Needs serverless GPU in the workspace, or a local NVIDIA GPU.

## 1. History in tapes

```bash
curl -fsSL https://raw.githubusercontent.com/pcc-labs/tapes-test/main/install.sh | sh
tapes-skills-demo check --ollama
```

With Go installed, `go install github.com/pcc-labs/tapes-test/cmd/tapes-skills-demo@latest`
works too. Releases before v0.1.3 read Codex only: when the person uses
Claude Code, confirm `check` prints a `claude code history` line, and if it
does not, the binary predates v0.1.3: run the install again.

On any `FAIL` line, stop and tell the user what it says. Docker not running
and a missing key are theirs to fix.

```bash
tapes-skills-demo --ollama                  # last 30 days; --since-days 0 for everything
```

This takes several minutes. Do not time it out. It ends by asking which
skills to write; that part is optional here, so tell the user to answer
`none`. The import is done once it prints `4/5 deriving`.

## 2. Repo ready

```bash
uv sync
databricks auth login --host https://<workspace>.cloud.databricks.com --profile tapes-eject
databricks warehouses list --profile tapes-eject
databricks catalogs list --profile tapes-eject
cp .env.example .env                        # only if .env does not exist yet
```

Ask the user which warehouse and catalog to use, then set
`TAPES_EJECT_CATALOG` and `DATABRICKS_WAREHOUSE_ID` in `.env`. The catalog
must allow creating a schema. `databricks auth login` opens a browser, so
the user runs it.

```bash
uv run tapes-eject doctor
```

Every line should say `ok`, except `labels: none yet` before step 3. On a
`FAIL` for Databricks auth, the warehouse, or the catalog, stop and tell the
user: those need their workspace access.

## 3. Labels

```bash
uv run tapes-eject label                    # 200 newest sessions
uv run tapes-eject label pushback --show 20
```

`label` matches patterns. It finds `pushback` (the person correcting the
agent) and `apology` (the agent backtracking), and prints the evidence for
each. **Read the evidence with the user.** Pattern matches include false
positives such as "dont send it yet" and plain "no".

```bash
uv run tapes-eject mark pushback <session-id> --remove     # drop a false positive
uv run tapes-eject mark golden <session-id> ...            # sessions worth learning from
uv run tapes-eject mark regression <session-id>            # sessions that went wrong
```

Labels live in `data/local_labels.jsonl`. A later `label` run replaces only
its own labels on the sessions it scans and never touches `mark`'s. Never
mark `golden` or `regression` on the user's behalf without asking.

## 4. Data

```bash
uv run tapes-eject export
uv run tapes-eject count
uv run tapes-eject sync
uv run tapes-eject report
uv run tapes-eject report --by project --label pushback
uv run tapes-eject report --by week
```

Text is redacted for common secret shapes, but redaction is pattern-based.
Before the first `sync`, tell the user that session text is uploaded to
their workspace, and suggest they skim `data/turns.jsonl`.

`sync` refuses a partial export, or one with no eval cases, because it would
delete good rows. Run `export` again. Pass `--force` only if the user accepts
the gaps.

When reporting results, give rates with their denominators ("3 of 40
sessions"), not rates alone: a model with few sessions swings wildly.

## 5. Fine-tune (optional)

Costs money. Confirm with the user before each run, and always run the
smoke run first.

On Databricks serverless GPU:

```bash
databricks bundle deploy
databricks bundle run sft_train --params max_steps=5
databricks bundle run eval_models --params limit=5
```

`RESOURCE_EXHAUSTED ... GPU quota ... is 0` means the workspace has no
serverless GPU (Free Edition never does). Offer the local path instead.

On a local NVIDIA GPU:

```bash
uv sync --group local
uv run --group local python train/local_sft.py --max-steps 5
uv run --group local python train/local_eval.py --limit 5
```

Full runs drop the step and case limits (`max_steps=0`, `limit=0`, or
`--max-steps 0` and no `--limit`). In MLflow, compare the `eval-base` and
`eval-tuned` runs. With few eval cases, say the comparison is weak.

`tapes-eject serve` creates a billed endpoint. Run it only when the user
asks, and run `tapes-eject unserve` when they are done.

## When something fails

| It says | Do this |
|---|---|
| `could not reach tapes` | Start Docker, then `tapes-skills-demo --ollama` again. It re-imports nothing. |
| `has no sessions` | `tapes-skills-demo check`; pass `--claude-root` or `--codex-root` if the history lives elsewhere. |
| No Claude Code sessions | The binary predates v0.1.3 (`check` shows no `claude code history` line). Install again, then import again. |
| `refusing to sync` | Read `data/report.json`, then run `export` again. |
| `only N training examples` | Import more history or `mark` more `golden` sessions, then `export` and `sync`. |
| `unknown resource` from `bundle run` | Run `databricks bundle deploy` first. |

## Cleanup

`uv run tapes-eject unserve` if an endpoint exists, and `uv run tapes-eject
spend` to check cost. `tapes-skills-demo down` deletes the local stack and its
data, never the history it read.
