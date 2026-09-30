# Live demo runbook (15 minutes)

Only Paper and Databricks appear on screen. Say "a supervised fine-tune of an open-weight model", never a tool name.

## The day before

- [ ] `uv run tapes-eject doctor`: all `ok`
- [ ] The autolabel cassette is running with `TYPESAFE_API_KEY` set
- [ ] `export` (fills `data/cache/`, so the live re-export is fast), `sync`, and the full `sft_train` and `eval_models` runs are done. Note the model version and both pass rates here: base ____ / tuned ____
  - No GPU quota in the workspace? Run `train/local_sft.py --max-steps 0` and `train/local_eval.py` on the local GPU instead (README, "Acts 3 and 4 on a local GPU"). The runs land in the same experiment.
- [ ] `uv run tapes-eject serve --version <v>`, then `ask` once to warm it. Leave it up until the demo.
  - Local GPU: `train/local_serve.py` in its own terminal, and `ask --url http://127.0.0.1:8081` in Act 4.
- [ ] `uv run tapes-eject spend`: total $____ of $400 (if `spend` cannot read the billing tables, take the number from the trial credit banner)
- [ ] Browser tabs: Paper console sessions list; Databricks Catalog → agent_sessions; MLflow experiment /Shared/tapes-eject; the endpoint page

## Act 1: Paper labels your agent work (4 min)

1. Console: the Labels column and filter. "Every session our agents run is captured and labeled."
2. Terminal: `uv run tapes-eject label pushback`. "It found where engineers pushed back, and wrote nothing."
3. `uv run tapes-eject label pushback --apply`, then refresh the console so the chips appear.
4. Mark one session `regression` in the console, live.

## Act 2: Databricks turns labels into data (4 min)

1. `uv run tapes-eject export && uv run tapes-eject count`. The full export ran the day before; this one reads unchanged sessions from `data/cache/` and only fetches what changed, including the session you just marked. Read out the counts. They are real numbers.
2. `uv run tapes-eject sync`
3. Catalog → `labels` → History: a new version from this sync. The `training_input` table holds the examples the labels selected; its lineage shows the volume file it was loaded from.
4. MLflow → Datasets → `eval_cases`: the regression you just marked is a new test case.

## Act 3: Train on Databricks GPUs (3 min, pre-run)

1. Jobs → `sft_train` → the finished full run: one H100, the loss curve.
2. Catalog → `agent_qwen3_4b`: versions and the run that produced each.

## Act 4: Prove it (3 min, pre-run)

1. MLflow → compare `eval-base` and `eval-tuned`: pass rates ____ vs ____.
2. Open one failed base case: the engineer's correction, the base answer, the tuned answer.
3. `uv run tapes-eject ask "<a prompt from a regression case>"` against the live endpoint.

## Close (1 min)

"Paper labels your agent work. Databricks turns the labels into datasets, evals, and models."

## After

- [ ] `uv run tapes-eject unserve`
- [ ] `uv run tapes-eject spend`: record the final total
