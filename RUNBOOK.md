# Live demo runbook (15 minutes)

Only Paper and Databricks appear on screen. Say "a supervised fine-tune of an open-weight model", never a tool name.

## The day before

Rules for anything that fetches sessions from Paper (`label`, `export`), learned the hard way (docs/postmortems/2026-09-30-paper-export-outages.md):

- Probe first: `paperctl sessions export <a tiny session id> | head -c 80`. If it does not answer, wait; do not start a run.
- One run at a time. Never a `label` run and an `export` at once.
- If a run reports "export service unreachable", stop. Do not retry it. It has already fallen back to the cache; the next run, later, fetches only what is missing.
- Run `export` the day before, not only on stage. The live `export` in Act 2 then reads from `data/cache/` and fetches only what changed.

- [ ] `uv run tapes-eject doctor`: all `ok`
- [ ] The autolabel cassette is running with `TYPESAFE_API_KEY` set
- [ ] `export` (fills `data/cache/`, so the live re-export is fast), `sync`, and the full `sft_train` and `eval_models` runs are done. Note the model version and both pass rates here: base ____ / tuned ____
  - No GPU quota in the workspace? Run `train/local_sft.py --max-steps 0` and `train/local_eval.py` on the local GPU instead (README, "Acts 3 and 4 on a local GPU"). The runs land in the same experiment.
- [ ] `uv run tapes-eject serve --version <v>`, then `ask` once to warm it. Leave it up until the demo.
  - Local GPU: `train/local_serve.py` in its own terminal, and `ask --url http://127.0.0.1:8081` in Act 4.
- [ ] `uv run tapes-eject spend`: total $____ of $400 (if `spend` cannot read the billing tables, take the number from the trial credit banner)
- [ ] Browser tabs: Paper console sessions list; Databricks Catalog → agent_sessions; MLflow experiment /Shared/tapes-eject; the endpoint page

## Act 1: Paper labels your agent work (3 min)

1. Console: the Labels column and filter. "Every session our agents run is captured and labeled."
2. Terminal: `uv run tapes-eject label pushback --sessions 1000`. "It found where engineers pushed back, and wrote nothing."
3. `... --apply`, then refresh the console so the chips appear.
4. Mark one session `regression` in the console, live.

## Act 2: Databricks turns labels into data (7 min)

1. `uv run tapes-eject export && uv run tapes-eject count`. The full export ran the day before; this one reads unchanged sessions from `data/cache/` and only fetches what changed, including the session you just marked. Read out the counts. They are real numbers.
2. `uv run tapes-eject sync`
3. Catalog → `labels` → History: a new version from this sync.
4. `uv run tapes-eject report`. "Which model gets corrected most." Then `--by project --label pushback`, then `--by week`.
5. SQL editor on `agent_sessions.corrections`: open two rows and read the engineer's words. These are the moments the labels captured.
6. If a dashboard is prepared: the same views, by model and by week.

## Act 3 (optional): The labels select the training data (3 min, pre-run)

1. `training_input` and the `eval_cases` dataset in MLflow: the labels chose both, and no session is in both.
2. MLflow → the `sft-...` run and Compare `eval-base` / `eval-tuned`: pass rates ____ vs ____. On a local GPU the runs carry a `-local` suffix.
3. If an endpoint is up: `uv run tapes-eject ask "<a prompt from a regression case>"`.

## Close (1 min)

"Paper labels your agent work. Databricks turns the labels into answers: which model, which project, which week, and, when you want it, a model trained on the sessions that went well."

## After

- [ ] `uv run tapes-eject unserve`
- [ ] `uv run tapes-eject spend`: record the final total
