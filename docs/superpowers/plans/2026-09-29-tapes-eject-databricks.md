# tapes-eject-databricks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A terminal-first demo that takes the labels Paper already holds, turns them into Unity Catalog tables and an MLflow evaluation dataset, fine-tunes Qwen3-4B on Databricks GPUs with the sessions the labels selected, and scores base against tuned.

**Architecture:** A small Python package, `tapes_eject`, runs on a laptop. It reads Paper through `paperctl`, since paperd supplies auth, and calls the autolabel cassette over HTTP. It writes redacted JSONL, uploads it to a Unity Catalog volume, and loads it with SQL on a SQL warehouse. It also builds the MLflow evaluation dataset. GPU work (training, evaluation, model registration) runs as Databricks notebook-source files on AI Runtime (serverless GPU) jobs. The laptop starts those jobs and creates the serving endpoint with the Databricks SDK.

**Tech Stack:** Python 3.11+, uv, `paperctl`, autolabel-cassette (`label_sampler`, pinned git dependency, used for session parsing and redaction), `databricks-sdk>=0.102.0`, `mlflow[databricks]>=3.12`, Databricks CLI, AI Runtime (AI environment v6+), TRL `SFTTrainer`, pytest.

**Spec:** `docs/specs/2026-09-29-databricks-labels-design.md`. Read it before starting.

## Global Constraints

- Everything runs from a fresh clone on a machine that is **not** the one this plan was written on. No step may depend on paths outside the repo, except `~/.databrickscfg`, `paperctl`'s own state, and the autolabel-cassette checkout the README tells you to make.
- Only two products appear on stage: Paper and Databricks. The fine-tuning code is described as "a supervised fine-tune (SFT) of an open-weight model on Databricks". Never name the training code.
- Labeling is Paper. Nothing in this repo writes labels to Paper except the live Act 1 command (`tapes-eject label --apply`), which goes through the autolabel cassette.
- paperd is the source of truth. Never read the autolabel-cassette repo's `labels/*.jsonl`.
- Never pass `--detail` to `paperctl sessions export`. `traces` strips spans.
- Unity Catalog names: schema `agent_sessions`. Tables `sessions`, `turns`, `labels`, `training_input`. MLflow evaluation dataset `eval_cases`. Volume `raw`. Model `<catalog>.agent_sessions.agent_qwen3_4b`. MLflow experiment `/Shared/tapes-eject`.
- Negative labels, which exclude a session from training: `pushback`, `apology`, `missing-knowledge`, `model-error`, `regression`. Correction labels, which become evaluation cases: `pushback`, `observation`, `missing-knowledge`.
- SFT only. No DPO or preference training. Do not use the Foundation Model Fine-tuning API (end-of-life).
- Budget: everything fits in the Databricks trial's $400. Every training run starts as a `max_steps=5` smoke run. Endpoints scale to zero and are deleted after the demo.
- Everything written to Databricks is redacted text. Raw export records are not uploaded (see "Deviations from the spec").

## Deviations from the spec (flagged for review)

1. **`sessions` has no `raw` column.** Raw export records carry unredacted tool output. Tables hold only the redacted text `label_sampler` produces.
2. **Training reads `training.jsonl` from the volume, not `training_input` through Spark.** AI Runtime notebooks are not guaranteed a Spark session. The table still exists for the Act 2 story. The run logs the table name as a parameter.
3. **Act 4 scores base and tuned inside a GPU job, not against two serving endpoints.** Only the tuned model gets an endpoint, for the live moment. This halves GPU serving spend.
4. **A session labeled both `golden` and `regression` is excluded from training.** A known failure never becomes training data.
5. **The optional "tuned endpoint proxied through tapes" close is not in this plan.**

## Review Focus

1. **A trace or span label whose session was never exported** (session too long, export failed, outside the sample). The label row is kept with `session_id = null` and counted in `data/report.json` as unmapped, not silently dropped. Tested in Task 4.
2. **A session export that fails or times out** (for example a 300-turn session). It is recorded in `report.json` under `failed`, and the rest of the export continues. Tested in Task 4.
3. **A label removed in Paper between syncs.** The row disappears from `labels` (`WHEN NOT MATCHED BY SOURCE THEN DELETE`), and the evaluation dataset is rebuilt without it. Tested in Task 6.
4. **Autolabel cassette unreachable during export.** `has_outcome` is `null` and those sessions are not training-eligible unless labeled `golden`. The export still completes, with a warning. Tested in Tasks 4 and 5.
5. **A correction label on a session's first turn, or the same turn labeled at both trace and span level.** The first-turn case is skipped because there is no earlier agent turn to judge, and duplicates collapse to one case. Tested in Task 5.

---

## File Structure

```
pyproject.toml                  uv project, deps, entry point, pytest config
.env.example                    every setting the CLI reads
databricks.yml                  bundle: syncs train/ to the workspace, holds the GPU jobs
resources/                      generated job YAML (Tasks 7-8)
src/tapes_eject/
  __init__.py
  config.py                     Config, load(), load_dotenv(), ping_url(), label constants
  doctor.py                     setup checks for `tapes-eject doctor`
  paper.py                      paperctl wrapper: labels, attachments, sessions, export
  autolabel.py                  autolabel cassette client: run + poll, matched sessions, evidence
  export.py                     Paper -> redacted rows + report; writes data/*.jsonl
  curate.py                     pure: training examples, eval records, counts
  databricks_io.py              SQL warehouse statements + volume uploads
  sync.py                       DDL/MERGE statements, upload order, eval dataset rebuild
  serve.py                      endpoint config, spend query
  cli.py                        argparse entry point: doctor, label, export, count, sync, serve, unserve, ask, spend
train/
  sft_train.py                  Databricks notebook source: SFT + register (GPU job)
  eval_models.py                Databricks notebook source: base vs tuned evaluation (GPU job)
tests/
  __init__.py
  helpers.py                    record() fixture builder, fakes
  test_config.py test_doctor.py test_paper.py test_autolabel.py
  test_export.py test_curate.py test_sync.py test_serve.py
README.md                       setup + each act, for a developer new to Databricks
RUNBOOK.md                      the live demo script
```

---

### Task 1: Scaffold, setup instructions, and `doctor`

**Files:**
- Create: `pyproject.toml`, `.env.example`, `src/tapes_eject/__init__.py`, `src/tapes_eject/config.py`, `src/tapes_eject/doctor.py`, `src/tapes_eject/cli.py`, `tests/__init__.py`, `tests/test_config.py`, `tests/test_doctor.py`
- Modify: `README.md` (add a Setup section), `.gitignore`

**Interfaces:**
- Produces: `config.Config` (fields `catalog, warehouse_id, profile, schema, volume, autolabel_url, org_slug, sample_sessions, max_turns, experiment, data_dir`; methods `table(name) -> str`, property `volume_path -> str`), `config.load(env: dict | None = None) -> Config`, `config.load_dotenv(path=Path(".env")) -> None`, `config.ping_url(base: str) -> str`, constants `NEGATIVE_LABELS: frozenset[str]`, `CORRECTION_LABELS: tuple[str, ...]`, `GOLDEN = "golden"`, `REGRESSION = "regression"`, `NO_OUTCOME = "no-outcome"`. `doctor.run_checks(checks) -> tuple[bool, list[str]]`. `cli.build_parser()` and `cli.main(argv=None) -> int`. Later tasks register subcommands in `build_parser()`.

- [ ] **Step 1: Set up the machine (manual, once)**

Run these on the machine that will run the demo. Record the values you get.

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# private pcc-labs git dependencies resolve through gh
gh auth login && gh auth setup-git
# Paper: log in and start paperd
paperctl login
paperctl init
paperctl status        # expect "auth: healthy" and your org
paperctl whoami        # note org_slug (papercomputeco is zro54)
# Databricks CLI
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks auth login --host https://<your-trial-workspace>.cloud.databricks.com --profile tapes-eject
databricks warehouses list --profile tapes-eject   # note the Serverless Starter Warehouse id
databricks catalogs list --profile tapes-eject      # pick the catalog you can create schemas in
# The autolabel cassette, in a sibling directory
git clone https://github.com/pcc-labs/autolabel-cassette ../autolabel-cassette
cd ../autolabel-cassette && uv sync
printf 'LABEL_SAMPLER_ORG=zro54\nTYPESAFE_API_KEY=<key>\n' > .env
uv run python -m label_sampler serve   # leave running: http://127.0.0.1:9996/v1/cassettes/autolabel
```

Check GPU access in the workspace UI. Go to New → Notebook → compute selector → **Serverless GPU**. In the Environment panel choose accelerator **1xH100** and environment **AI v6**, then Apply, and run a cell with `%sh nvidia-smi`.
Expected: one `NVIDIA H100 80GB HBM3`. If H100 is not offered, choose **A10**, note it, and use the LoRA path in Task 7.

- [ ] **Step 2: Write `pyproject.toml`, `.env.example`, `.gitignore`**

`pyproject.toml`:
```toml
[project]
name = "tapes-eject-databricks"
version = "0.1.0"
description = "Paper labels to Databricks datasets, evals, and a fine-tuned model"
requires-python = ">=3.11"
dependencies = [
  "autolabel-cassette",
  "databricks-sdk>=0.102.0",
  "mlflow[databricks]>=3.12",
]

[project.scripts]
tapes-eject = "tapes_eject.cli:main"

[dependency-groups]
dev = ["pytest", "ruff"]

[tool.uv.sources]
autolabel-cassette = { git = "https://github.com/pcc-labs/autolabel-cassette", rev = "1a43a65b969a40cc41892f451a13850905e919cb" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tapes_eject"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "."]

[tool.ruff]
line-length = 100
target-version = "py311"
```

`.env.example`:
```bash
# Unity Catalog catalog you can create a schema in (databricks catalogs list)
TAPES_EJECT_CATALOG=
# SQL warehouse id (databricks warehouses list)
DATABRICKS_WAREHOUSE_ID=
# Databricks CLI profile from `databricks auth login --profile`
DATABRICKS_CONFIG_PROFILE=tapes-eject
# Autolabel cassette base, including its route prefix
AUTOLABEL_URL=http://127.0.0.1:9996/v1/cassettes/autolabel
# Paper org slug; empty uses paperctl's active org
PAPER_ORG_SLUG=
# How many recent unlabeled sessions to sample as training candidates
TAPES_EJECT_SAMPLE_SESSIONS=200
# Sessions longer than this are not exported
TAPES_EJECT_MAX_TURNS=150
```

Append to `.gitignore`: `.venv/`, `*.egg-info/`, `resources/*.src/` (the existing `data/`, `.databricks/`, `.env` lines stay).

- [ ] **Step 3: Write the failing tests**

`tests/__init__.py`: empty file.

`tests/test_config.py`:
```python
import pytest

from tapes_eject.config import load, ping_url

BASE = {"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh1"}


def test_load_requires_catalog_and_warehouse():
    with pytest.raises(SystemExit, match="TAPES_EJECT_CATALOG"):
        load({"DATABRICKS_WAREHOUSE_ID": "wh1"})
    with pytest.raises(SystemExit, match="DATABRICKS_WAREHOUSE_ID"):
        load({"TAPES_EJECT_CATALOG": "demo"})


def test_load_defaults_and_names():
    cfg = load({**BASE, "AUTOLABEL_URL": "http://h:9996/v1/cassettes/autolabel/"})
    assert cfg.table("labels") == "demo.agent_sessions.labels"
    assert cfg.volume_path == "/Volumes/demo/agent_sessions/raw"
    assert cfg.autolabel_url == "http://h:9996/v1/cassettes/autolabel"
    assert cfg.profile == "tapes-eject"
    assert cfg.sample_sessions == 200 and cfg.max_turns == 150
    assert cfg.experiment == "/Shared/tapes-eject"
    assert cfg.org_slug is None


def test_ping_url_is_the_host_root():
    assert ping_url("http://127.0.0.1:9996/v1/cassettes/autolabel") == "http://127.0.0.1:9996/ping"
    assert ping_url("https://box.example/api/autolabel") == "https://box.example/ping"
```

`tests/test_doctor.py`:
```python
from tapes_eject.doctor import run_checks


def test_run_checks_reports_each_and_fails_on_any_error():
    def boom():
        raise RuntimeError("paperd not running")

    ok, lines = run_checks([("a", lambda: "fine"), ("b", boom)])
    assert ok is False
    assert lines == ["ok    a: fine", "FAIL  b: paperd not running"]


def test_run_checks_all_pass():
    ok, _ = run_checks([("a", lambda: "x")])
    assert ok is True
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv sync && uv run pytest tests/test_config.py tests/test_doctor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tapes_eject'`

- [ ] **Step 5: Implement `config.py`, `doctor.py`, `cli.py`**

`src/tapes_eject/__init__.py`:
```python
"""Paper labels to Databricks datasets, evals, and a fine-tuned model."""
```

`src/tapes_eject/config.py`:
```python
"""Settings, read from the environment. The CLI loads `.env` first."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

NEGATIVE_LABELS = frozenset({"pushback", "apology", "missing-knowledge", "model-error", "regression"})
CORRECTION_LABELS = ("pushback", "observation", "missing-knowledge")
GOLDEN = "golden"
REGRESSION = "regression"
NO_OUTCOME = "no-outcome"

DEFAULT_AUTOLABEL_URL = "http://127.0.0.1:9996/v1/cassettes/autolabel"


@dataclass(frozen=True)
class Config:
    catalog: str
    warehouse_id: str
    profile: str = "tapes-eject"
    schema: str = "agent_sessions"
    volume: str = "raw"
    autolabel_url: str = DEFAULT_AUTOLABEL_URL
    org_slug: str | None = None
    sample_sessions: int = 200
    max_turns: int = 150
    experiment: str = "/Shared/tapes-eject"
    data_dir: Path = Path("data")

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"


def load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=value lines into the environment. Variables already set win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load(env: dict[str, str] | None = None) -> Config:
    env = dict(os.environ) if env is None else env
    missing = [k for k in ("TAPES_EJECT_CATALOG", "DATABRICKS_WAREHOUSE_ID") if not env.get(k)]
    if missing:
        raise SystemExit(f"missing settings: {', '.join(missing)} (copy .env.example to .env)")
    return Config(
        catalog=env["TAPES_EJECT_CATALOG"],
        warehouse_id=env["DATABRICKS_WAREHOUSE_ID"],
        profile=env.get("DATABRICKS_CONFIG_PROFILE") or "tapes-eject",
        autolabel_url=(env.get("AUTOLABEL_URL") or DEFAULT_AUTOLABEL_URL).rstrip("/"),
        org_slug=env.get("PAPER_ORG_SLUG") or None,
        sample_sessions=int(env.get("TAPES_EJECT_SAMPLE_SESSIONS") or 200),
        max_turns=int(env.get("TAPES_EJECT_MAX_TURNS") or 150),
    )


def ping_url(base: str) -> str:
    """The cassette answers /ping at its host root, whatever its route prefix."""
    parts = urlsplit(base)
    return f"{parts.scheme}://{parts.netloc}/ping"
```

`src/tapes_eject/doctor.py`:
```python
"""`tapes-eject doctor`: is this machine ready to run the demo?"""

from __future__ import annotations

import functools
import json
import subprocess
import urllib.request
from typing import Callable

from .config import Config, ping_url

Check = tuple[str, Callable[[], str]]


def run_checks(checks: list[Check]) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for name, fn in checks:
        try:
            lines.append(f"ok    {name}: {fn()}")
        except Exception as e:  # a check's failure is its message, whatever raised it
            ok = False
            lines.append(f"FAIL  {name}: {e}")
    return ok, lines


def _paperctl(*argv: str) -> str:
    proc = subprocess.run(["paperctl", *argv], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"paperctl {argv[0]} failed")
    return proc.stdout


def _paperd() -> str:
    out = _paperctl("status")
    if "healthy" not in out:
        raise RuntimeError("paperd auth is not healthy: run `paperctl login`, then `paperctl init`")
    return "running, auth healthy"


def _paper_org() -> str:
    for line in _paperctl("whoami").splitlines():
        if line.startswith("org_slug:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("`paperctl whoami` printed no org_slug")


def _autolabel(cfg: Config) -> str:
    with urllib.request.urlopen(ping_url(cfg.autolabel_url), timeout=5) as resp:
        ping = json.loads(resp.read())
    org = cfg.org_slug or _paper_org()
    if ping.get("org") != org:
        raise RuntimeError(f"cassette org is {ping.get('org')!r}, Paper org is {org!r}: set LABEL_SAMPLER_ORG")
    judge = "on" if ping.get("judge") else "OFF (pushback/question/observation answer needs_judge)"
    return f"org={org} judge={judge}"


@functools.lru_cache(maxsize=None)
def _workspace(profile: str):
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile=profile)


def _mlflow_datasets() -> str:
    import mlflow
    import mlflow.genai.datasets as ds

    for name in ("create_dataset", "get_dataset", "delete_dataset"):
        if not hasattr(ds, name):
            raise RuntimeError(f"mlflow {mlflow.__version__} has no genai.datasets.{name}; need >=3.12")
    return f"mlflow {mlflow.__version__}"


def checks(cfg: Config) -> list[Check]:
    w = lambda: _workspace(cfg.profile)  # noqa: E731
    return [
        ("paperd", _paperd),
        ("paper org", lambda: cfg.org_slug or _paper_org()),
        ("autolabel cassette", lambda: _autolabel(cfg)),
        ("databricks auth", lambda: w().current_user.me().user_name),
        ("sql warehouse", lambda: str(w().warehouses.get(cfg.warehouse_id).state)),
        ("catalog", lambda: w().catalogs.get(cfg.catalog).name),
        ("mlflow eval datasets", _mlflow_datasets),
    ]
```

`src/tapes_eject/cli.py`:
```python
"""tapes-eject: Paper labels to Databricks datasets, evals, and a fine-tuned model."""

from __future__ import annotations

import argparse
import sys

from . import config, doctor


def cmd_doctor(cfg: config.Config, args: argparse.Namespace) -> int:
    ok, lines = doctor.run_checks(doctor.checks(cfg))
    print("\n".join(lines))
    if not ok:
        print("\nFix the FAIL lines; README 'Setup' covers each one.")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tapes-eject")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check paperd, the cassette, and the workspace").set_defaults(
        fn=cmd_doctor
    )
    return p


def main(argv: list[str] | None = None) -> int:
    config.load_dotenv()
    args = build_parser().parse_args(argv)
    return args.fn(config.load(), args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_doctor.py -v`
Expected: 5 passed

- [ ] **Step 7: Add the README Setup section and run doctor for real**

Append to `README.md`:
````markdown
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
````

Run: `uv run tapes-eject doctor`
Expected: seven `ok` lines. Fix any `FAIL` before continuing.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .env.example .gitignore README.md src tests
git commit -m "Scaffold tapes-eject with settings and a doctor command"
```

---

### Task 2: Read Paper through `paperctl`

**Files:**
- Create: `src/tapes_eject/paper.py`, `tests/helpers.py`, `tests/test_paper.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `paper.PaperError(RuntimeError)`, `paper.Runner = Callable[[list[str], int], str]`, `paper.paperctl(argv: list[str], timeout: int) -> str`, and class `paper.Paper(run: Runner = paperctl, org_slug: str | None = None)` with methods `labels() -> list[dict]` (each has `id`, `name`, `usage`), `attachments(label_id: str, primitive_type: str) -> list[dict]` (each has `primitive_type`, `primitive_id`), `sessions(label: str | None = None, since: str | None = None, limit: int = 200) -> list[dict]` (each has `id`, `rollup.turn_count`), and `export_session(session_id: str, timeout: int = 300) -> dict | None`. `tests/helpers.py` provides `FakeRunner` and `record(...)`.

- [ ] **Step 1: Write the test helpers**

`tests/helpers.py`:
```python
"""Shared fixtures: a scripted paperctl and minimal export records."""

from __future__ import annotations

import json


class FakeRunner:
    """Answers paperctl argv by the first rule whose tokens all appear in it."""

    def __init__(self, rules: list[tuple[tuple[str, ...], object]]):
        self.rules = rules
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], timeout: int) -> str:
        self.calls.append(argv)
        for tokens, answer in self.rules:
            if all(t in argv for t in tokens):
                if isinstance(answer, Exception):
                    raise answer
                return answer if isinstance(answer, str) else json.dumps(answer)
        raise AssertionError(f"unexpected paperctl call: {argv}")


def record(session_id: str, turns: list[tuple[str, str, str]], title: str = "t") -> dict:
    """An export record. turns: (trace_id, user_prompt, assistant_text). Each turn has one llm span
    whose id is `spn_<trace_id>`."""
    return {
        "schema": "2026-06-15",
        "session": {
            "id": session_id,
            "harness_id": "claude",
            "display_title": title,
            "started_at": "2026-09-01T00:00:00Z",
            "rollup": {
                "status": "completed",
                "turn_count": len(turns),
                "model": "claude-opus-5-5",
                "usage": {"cost_usd": 1.5},
            },
        },
        "traces": [
            {
                "trace": {
                    "trace_id": tid,
                    "user_prompt": prompt,
                    "response_preview": reply[:40],
                    "status": "ok",
                    "started_at": f"2026-09-01T00:{i:02d}:00Z",
                },
                "spans": [
                    {
                        "span_id": f"spn_{tid}",
                        "kind": "llm",
                        "seq": 0,
                        "output": [{"type": "text", "text": reply}],
                    }
                ],
            }
            for i, (tid, prompt, reply) in enumerate(turns)
        ],
    }
```

- [ ] **Step 2: Write the failing tests**

`tests/test_paper.py`:
```python
import json
import subprocess

import pytest

from tapes_eject import paper
from tapes_eject.paper import Paper, PaperError
from tests.helpers import FakeRunner, record


def test_labels_parses_the_list():
    run = FakeRunner([(("list-labels",), {"labels": [{"id": "L1", "name": "pushback"}]})])
    assert Paper(run).labels() == [{"id": "L1", "name": "pushback"}]


def test_attachments_follow_the_cursor_to_the_end():
    run = FakeRunner(
        [
            (("--cursor", "c1"), {"attachments": [{"primitive_id": "b"}], "next_cursor": None}),
            (("list-label-attachments",), {"attachments": [{"primitive_id": "a"}], "next_cursor": "c1"}),
        ]
    )
    got = Paper(run).attachments("L1", "trace")
    assert [a["primitive_id"] for a in got] == ["a", "b"]
    assert run.calls[0][-4:] == ["--primitive-type", "trace", "--limit", "200"]


def test_sessions_passes_label_json_and_pages_until_limit():
    run = FakeRunner(
        [
            (("--cursor", "n1"), {"items": [{"id": "s3"}], "next_cursor": None}),
            (("sessions", "list"), {"items": [{"id": "s1"}, {"id": "s2"}], "next_cursor": "n1"}),
        ]
    )
    got = Paper(run).sessions(label="golden", limit=3)
    assert [s["id"] for s in got] == ["s1", "s2", "s3"]
    assert "--json" in run.calls[0] and ["--label", "golden"] == run.calls[0][-2:]


def test_export_never_passes_detail_and_returns_the_record():
    rec = record("s1", [("trc_1", "hi", "hello")])
    run = FakeRunner([(("sessions", "export", "s1"), json.dumps(rec) + "\n")])
    assert Paper(run).export_session("s1") == rec
    assert "--detail" not in run.calls[0]


def test_org_slug_goes_first():
    run = FakeRunner([(("list-labels",), {"labels": []})])
    Paper(run, org_slug="zro54").labels()
    assert run.calls[0][:2] == ["--org-slug", "zro54"]


def test_paperctl_timeout_becomes_paper_error(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="paperctl", timeout=1)

    monkeypatch.setattr(paper.subprocess, "run", slow)
    with pytest.raises(PaperError, match="timed out"):
        paper.paperctl(["sessions", "export", "s1"], 1)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_paper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tapes_eject.paper'`

- [ ] **Step 4: Implement `paper.py`**

`src/tapes_eject/paper.py`:
```python
"""Read Paper through paperctl. paperd supplies auth and routing; this module holds no token."""

from __future__ import annotations

import json
import subprocess
from typing import Callable

Runner = Callable[[list[str], int], str]


class PaperError(RuntimeError):
    pass


def paperctl(argv: list[str], timeout: int) -> str:
    try:
        proc = subprocess.run(
            ["paperctl", *argv], capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise PaperError(f"paperctl {' '.join(argv)} timed out after {timeout}s") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[:500]
        raise PaperError(f"paperctl {' '.join(argv)} exited {proc.returncode}: {detail}")
    return proc.stdout


class Paper:
    def __init__(self, run: Runner = paperctl, org_slug: str | None = None):
        self._run = run
        self._org = org_slug

    def _call(self, argv: list[str], timeout: int = 120) -> str:
        prefix = ["--org-slug", self._org] if self._org else []
        return self._run([*prefix, *argv], timeout)

    def labels(self) -> list[dict]:
        return json.loads(self._call(["cassettes", "labels", "list-labels"]))["labels"]

    def attachments(self, label_id: str, primitive_type: str) -> list[dict]:
        out: list[dict] = []
        cursor: str | None = None
        while True:
            argv = ["cassettes", "labels", "list-label-attachments", label_id]
            if cursor:
                argv += ["--cursor", cursor]
            argv += ["--primitive-type", primitive_type, "--limit", "200"]
            page = json.loads(self._call(argv))
            out.extend(page.get("attachments") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                return out

    def sessions(
        self, label: str | None = None, since: str | None = None, limit: int = 200
    ) -> list[dict]:
        """Session list items, newest first, paged until `limit`."""
        out: list[dict] = []
        cursor: str | None = None
        while len(out) < limit:
            argv = ["sessions", "list", "--json", "--limit", str(min(200, limit - len(out)))]
            if cursor:
                argv += ["--cursor", cursor]
            if since:
                argv += ["--since", since]
            if label:
                argv += ["--label", label]
            page = json.loads(self._call(argv))
            items = page.get("items") or []
            out.extend(items)
            cursor = page.get("next_cursor")
            if not cursor or not items:
                break
        return out[:limit]

    def export_session(self, session_id: str, timeout: int = 300) -> dict | None:
        """The full record at paperctl's default detail. Never pass --detail: `traces` strips spans."""
        text = self._call(["sessions", "export", session_id], timeout)
        for line in text.splitlines():
            if line.strip():
                return json.loads(line)
        return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_paper.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/tapes_eject/paper.py tests/helpers.py tests/test_paper.py
git commit -m "Read labels, attachments, sessions, and exports through paperctl"
```

---

### Task 3: The autolabel cassette client and the live `label` command

**Files:**
- Create: `src/tapes_eject/autolabel.py`, `tests/test_autolabel.py`
- Modify: `src/tapes_eject/cli.py`

**Interfaces:**
- Consumes: `config.ping_url`, `config.Config`, `paper.Paper`.
- Produces: `autolabel.AutolabelError(RuntimeError)`, `autolabel.Transport = Callable[[str, str, dict | None], dict]`, `autolabel.http_json(method, url, body=None, timeout=30) -> dict`, and class `autolabel.Autolabel(base: str, transport: Transport = http_json, poll_seconds: float = 1.5, max_wait_seconds: float = 1800, sleep=time.sleep)` with methods `ping() -> dict`, `run(label: str, session_ids: list[str], apply: bool = False, on_progress=None) -> dict` (the cassette's `result`), `matched_sessions(label: str, session_ids: list[str], chunk: int = 25) -> set[str]`, and `turn_evidence(label: str, session_ids: list[str], chunk: int = 25) -> tuple[dict[str, str], str | None]` (trace id → evidence, plus the cassette's `reason`, e.g. `needs_judge`). CLI: `tapes-eject label <name> [--sessions N] [--apply]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_autolabel.py`:
```python
import pytest

from tapes_eject.autolabel import Autolabel, AutolabelError

BASE = "http://h:9996/v1/cassettes/autolabel"


def scripted(*answers):
    calls = []
    queue = list(answers)

    def transport(method, url, body):
        calls.append((method, url, body))
        return queue.pop(0)

    return transport, calls


def result(sessions, reason=None):
    return {"label": "x", "sessions": sessions, "reason": reason, "matched": 0}


def test_run_posts_then_polls_until_done():
    t, calls = scripted(
        {"id": "r1", "state": "running", "progress": "reading sessions"},
        {"id": "r1", "state": "running", "progress": "scanning 2 sessions"},
        {"id": "r1", "state": "done", "result": result([])},
    )
    seen = []
    got = Autolabel(BASE, t, sleep=lambda s: None).run("apology", ["s1"], on_progress=seen.append)
    assert got["sessions"] == []
    assert calls[0] == ("POST", f"{BASE}/run", {"label": "apology", "session_ids": ["s1"], "apply": False})
    assert calls[1][:2] == ("GET", f"{BASE}/runs/r1")
    assert seen == ["scanning 2 sessions"]


def test_run_raises_when_the_job_failed():
    t, _ = scripted({"id": "r1", "state": "running"}, {"id": "r1", "state": "failed", "error": "boom"})
    with pytest.raises(AutolabelError, match="failed: boom"):
        Autolabel(BASE, t, sleep=lambda s: None).run("apology", ["s1"])


def test_run_gives_up_after_max_wait():
    t, _ = scripted(*[{"id": "r1", "state": "running"}] * 5)
    client = Autolabel(BASE, t, poll_seconds=1, max_wait_seconds=2, sleep=lambda s: None)
    with pytest.raises(AutolabelError, match="still running"):
        client.run("apology", ["s1"])


def test_matched_sessions_chunks_and_unions():
    t, calls = scripted(
        {"id": "a", "state": "done", "result": result([{"session_id": "s1", "matched": True, "turns": []}, {"session_id": "s2", "matched": False, "turns": []}])},
        {"id": "b", "state": "done", "result": result([{"session_id": "s3", "matched": True, "turns": []}])},
    )
    got = Autolabel(BASE, t, sleep=lambda s: None).matched_sessions("no-outcome", ["s1", "s2", "s3"], chunk=2)
    assert got == {"s1", "s3"}
    assert [c[2]["session_ids"] for c in calls] == [["s1", "s2"], ["s3"]]


def test_turn_evidence_maps_trace_to_evidence_and_passes_reason():
    turns = [{"turn_id": "trc_1", "evidence": "no, use the other flag"}, {"turn_id": "trc_1", "evidence": "dup"}]
    t, _ = scripted(
        {"id": "a", "state": "done", "result": result([{"session_id": "s1", "matched": True, "turns": turns}], reason="needs_judge")}
    )
    ev, reason = Autolabel(BASE, t, sleep=lambda s: None).turn_evidence("pushback", ["s1"])
    assert ev == {"trc_1": "no, use the other flag"}
    assert reason == "needs_judge"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_autolabel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tapes_eject.autolabel'`

- [ ] **Step 3: Implement `autolabel.py`**

`src/tapes_eject/autolabel.py`:
```python
"""Client for the autolabel cassette: one run per label, find first, apply only when asked.

POST {base}/run answers 202 with a job; GET {base}/runs/<id> is polled until the state leaves
`running`. The result lists each requested session with the turns the label sits on.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable

from .config import ping_url

Transport = Callable[[str, str, "dict | None"], dict]


class AutolabelError(RuntimeError):
    pass


def http_json(method: str, url: str, body: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise AutolabelError(f"{method} {url}: HTTP {e.code} {e.read()[:300]!r}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AutolabelError(f"{method} {url}: unreachable ({e})") from e


class Autolabel:
    def __init__(
        self,
        base: str,
        transport: Transport = http_json,
        poll_seconds: float = 1.5,
        max_wait_seconds: float = 1800,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base = base.rstrip("/")
        self._t = transport
        self._poll = poll_seconds
        self._max_wait = max_wait_seconds
        self._sleep = sleep

    def ping(self) -> dict:
        return self._t("GET", ping_url(self.base), None)

    def run(
        self,
        label: str,
        session_ids: list[str],
        apply: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict:
        body = {"label": label, "session_ids": list(session_ids), "apply": apply}
        job = self._t("POST", f"{self.base}/run", body)
        waited = 0.0
        while job.get("state", "running") == "running":
            if waited >= self._max_wait:
                raise AutolabelError(f"run {job.get('id')} still running after {waited:.0f}s")
            self._sleep(self._poll)
            waited += self._poll
            job = self._t("GET", f"{self.base}/runs/{job['id']}", None)
            if on_progress and job.get("progress"):
                on_progress(job["progress"])
        if job["state"] != "done":
            raise AutolabelError(f"run {job.get('id')} {job['state']}: {job.get('error', '')}")
        return job["result"]

    def matched_sessions(self, label: str, session_ids: list[str], chunk: int = 25) -> set[str]:
        hit: set[str] = set()
        for i in range(0, len(session_ids), chunk):
            res = self.run(label, session_ids[i : i + chunk])
            hit |= {s["session_id"] for s in res["sessions"] if s["matched"]}
        return hit

    def turn_evidence(
        self, label: str, session_ids: list[str], chunk: int = 25
    ) -> tuple[dict[str, str], str | None]:
        evidence: dict[str, str] = {}
        reason: str | None = None
        for i in range(0, len(session_ids), chunk):
            res = self.run(label, session_ids[i : i + chunk])
            reason = reason or res.get("reason")
            for s in res["sessions"]:
                for turn in s.get("turns") or []:
                    if turn.get("turn_id") and turn.get("evidence"):
                        evidence.setdefault(turn["turn_id"], turn["evidence"])
        return evidence, reason
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_autolabel.py -v`
Expected: 5 passed

- [ ] **Step 5: Add the `label` command (Act 1 on stage)**

In `src/tapes_eject/cli.py`, add these imports under `from . import config, doctor`:
```python
from .autolabel import Autolabel
from .paper import Paper
```

Add this function above `build_parser`:
```python
def cmd_label(cfg: config.Config, args: argparse.Namespace) -> int:
    """Find a label across the newest sessions; with --apply, label them in Paper."""
    ids = [s["id"] for s in Paper(org_slug=cfg.org_slug).sessions(limit=args.sessions)]
    client = Autolabel(cfg.autolabel_url)
    res = client.run(args.name, ids, apply=False, on_progress=lambda p: print(f"  {p}"))
    if res.get("reason"):
        print(f"{args.name}: {res['reason']}")
        return 1
    print(f"{args.name} found in {res['matched']} of {len(ids)} sessions")
    if not args.apply or not res["matched"]:
        return 0
    done = client.run(args.name, ids, apply=True, on_progress=lambda p: print(f"  {p}"))
    print(f"{args.name}: {done['created']} new labels, {done['pushed']} pushed to Paper")
    return 0 if not done["push_failed"] else 1
```

In `build_parser`, before `return p`:
```python
    lab = sub.add_parser("label", help="Act 1: find a label across recent sessions, then apply it")
    lab.add_argument("name", help="apology, dream, subagents, no-outcome, pushback, question, observation")
    lab.add_argument("--sessions", type=int, default=25)
    lab.add_argument("--apply", action="store_true", help="write the labels to Paper")
    lab.set_defaults(fn=cmd_label)
```

- [ ] **Step 6: Try the find against the real cassette (writes nothing)**

Run: `uv run tapes-eject label apology --sessions 25`
Expected: progress lines, then `apology found in N of 25 sessions`. Do not pass `--apply` yet. Applying is the live demo moment.

- [ ] **Step 7: Commit**

```bash
git add src/tapes_eject/autolabel.py src/tapes_eject/cli.py tests/test_autolabel.py
git commit -m "Call the autolabel cassette and add the live label command"
```

---

### Task 4: Export labeled sessions from Paper

**Files:**
- Create: `src/tapes_eject/export.py`, `tests/test_export.py`
- Modify: `src/tapes_eject/cli.py`

**Interfaces:**
- Consumes: `paper.Paper`, `paper.PaperError`, `autolabel.Autolabel`, `autolabel.AutolabelError`, `config.Config`, `config.CORRECTION_LABELS`, `config.NO_OUTCOME`, and `label_sampler.session.parse_session(record) -> Session | None` (its `Session` has `id, title, harness, model, started_at, status, turn_count, cost_usd, turns`, and each `Turn` has `id, index, user_prompt, response_preview, assistant_text: list[str], synthetic`; the text is already redacted).
- Produces: `export.LEVELS = ("session", "trace", "span")`; `export.index_record(record) -> tuple[str, dict[str, str], dict[str, str]]` (session id, trace→session, span→trace); `export.label_rows(attachments: dict[str, list[dict]], trace_session: dict, span_trace: dict, evidence: dict[tuple[str, str], str] | None = None) -> tuple[list[dict], list[dict]]` (rows, unmapped); `export.session_row(sess, no_outcome: set[str] | None) -> dict`; `export.turn_rows(sess) -> list[dict]`; `export.choose_sessions(labeled: dict[str, dict], recent: list[dict], sample: int, max_turns: int) -> tuple[list[str], list[tuple[str, str]]]`; dataclass `export.Export(sessions, turns, labels, failed, unmapped, outcome_known)`; `export.run_export(paper, autolabel, cfg, with_evidence=False, log=print) -> Export`; `export.write_export(ex: Export, data_dir: Path) -> None`, which writes `sessions.jsonl`, `turns.jsonl`, `labels.jsonl`, `report.json`. Row schemas:
  - session: `session_id, title, harness, model, started_at, status, turn_count, cost_usd, has_outcome (bool | None)`
  - turn: `session_id, turn_id, ordinal, user_prompt, agent_text, synthetic`
  - label: `label, primitive_type, primitive_id, session_id, turn_id, span_id, evidence`

- [ ] **Step 1: Write the failing tests**

`tests/test_export.py`:
```python
import json

from label_sampler.session import parse_session

from tapes_eject.autolabel import AutolabelError
from tapes_eject.config import load
from tapes_eject.export import choose_sessions, label_rows, run_export, turn_rows, write_export
from tapes_eject.paper import PaperError
from tests.helpers import record

CFG = load({"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh", "TAPES_EJECT_SAMPLE_SESSIONS": "5"})


class FakePaper:
    def __init__(self, labels, attachments, by_label, recent, records, fail=()):
        self._labels, self._att, self._by_label = labels, attachments, by_label
        self._recent, self._records, self._fail = recent, records, set(fail)

    def labels(self):
        return self._labels

    def attachments(self, label_id, primitive_type):
        return self._att.get((label_id, primitive_type), [])

    def sessions(self, label=None, since=None, limit=200):
        return self._by_label.get(label, []) if label else self._recent

    def export_session(self, session_id, timeout=300):
        if session_id in self._fail:
            raise PaperError("paperctl sessions export timed out after 300s")
        return self._records.get(session_id)


class FakeAutolabel:
    def __init__(self, no_outcome=(), error=None):
        self.no_outcome, self.error = set(no_outcome), error

    def matched_sessions(self, label, ids, chunk=25):
        if self.error:
            raise self.error
        return self.no_outcome & set(ids)

    def turn_evidence(self, label, ids, chunk=25):
        return {}, None


def item(sid, turns=3):
    return {"id": sid, "rollup": {"turn_count": turns}}


def att(ptype, pid):
    return {"primitive_type": ptype, "primitive_id": pid}


def test_label_rows_map_trace_and_span_to_their_session():
    rows, unmapped = label_rows(
        {"pushback": [att("trace", "trc_1"), att("span", "spn_trc_1"), att("session", "s1")]},
        trace_session={"trc_1": "s1"},
        span_trace={"spn_trc_1": "trc_1"},
    )
    assert [(r["primitive_type"], r["session_id"], r["turn_id"]) for r in rows] == [
        ("trace", "s1", "trc_1"),
        ("span", "s1", "trc_1"),
        ("session", "s1", None),
    ]
    assert unmapped == []


def test_unmapped_attachment_is_kept_with_null_session_and_reported():
    rows, unmapped = label_rows({"pushback": [att("trace", "trc_gone")]}, {}, {})
    assert rows == unmapped
    assert rows[0]["session_id"] is None and rows[0]["turn_id"] == "trc_gone"


def test_label_rows_skip_non_session_primitives():
    rows, _ = label_rows({"paper": [att("skill", "sk_1")]}, {}, {})
    assert rows == []


def test_choose_sessions_skips_oversized_labeled_and_samples_short_recent():
    labeled = {"big": item("big", 300), "s1": item("s1", 4)}
    recent = [item("s1"), item("r1", 1), item("r2", 10), item("r3", 60), item("r4", 5)]
    ids, skipped = choose_sessions(labeled, recent, sample=1, max_turns=150)
    assert ids == ["s1", "r2"]
    assert skipped == [("big", "300 turns > max 150")]


def test_turn_rows_are_redacted():
    sess = parse_session(record("s1", [("trc_1", "use key sk-ant-abcdefghijklmnopqrstuvwxyz0123", "ok")]))
    (row,) = turn_rows(sess)
    assert "sk-ant-" not in row["user_prompt"] and "[redacted]" in row["user_prompt"]
    assert row["agent_text"] == "ok"


def _paper(fail=()):
    return FakePaper(
        labels=[{"id": "L1", "name": "pushback", "usage": {"session": 1, "trace": 2}}],
        attachments={
            ("L1", "session"): [att("session", "s1")],
            ("L1", "trace"): [att("trace", "trc_b"), att("trace", "trc_elsewhere")],
        },
        by_label={"pushback": [item("s1")]},
        recent=[item("s1"), item("s2"), item("s3")],
        records={
            "s1": record("s1", [("trc_a", "do x", "did x"), ("trc_b", "no, do y", "did y")]),
            "s2": record("s2", [("trc_c", "hello", "hi"), ("trc_d", "thanks", "np")]),
        },
        fail=fail,
    )


def test_run_export_continues_past_a_failed_export_and_reports_it(tmp_path):
    ex = run_export(_paper(fail={"s3"}), FakeAutolabel(no_outcome={"s2"}), CFG, log=lambda m: None)
    assert {s["session_id"] for s in ex.sessions} == {"s1", "s2"}
    assert ex.failed == [("s3", "paperctl sessions export timed out after 300s")]
    assert {s["session_id"]: s["has_outcome"] for s in ex.sessions} == {"s1": True, "s2": False}
    assert [u["primitive_id"] for u in ex.unmapped] == ["trc_elsewhere"]
    write_export(ex, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failed"] == [["s3", "paperctl sessions export timed out after 300s"]]
    assert report["unmapped"] == 1 and report["outcome_known"] is True
    assert len((tmp_path / "labels.jsonl").read_text().splitlines()) == 3


def test_run_export_marks_outcome_unknown_when_the_cassette_is_down():
    down = FakeAutolabel(error=AutolabelError("POST .../run: unreachable"))
    ex = run_export(_paper(), down, CFG, log=lambda m: None)
    assert ex.outcome_known is False
    assert all(s["has_outcome"] is None for s in ex.sessions)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tapes_eject.export'`

- [ ] **Step 3: Implement `export.py`**

`src/tapes_eject/export.py`:
```python
"""Paper -> redacted rows. Labels come from paperd; outcomes from the autolabel cassette.

Session text is parsed and redacted by label_sampler (the autolabel cassette's own parser), so
nothing written here carries a secret-shaped string. Raw records are never written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from label_sampler.session import Session, parse_session

from .autolabel import AutolabelError
from .config import CORRECTION_LABELS, NO_OUTCOME, Config
from .paper import PaperError

LEVELS = ("session", "trace", "span")
RECENT_MIN_TURNS, RECENT_MAX_TURNS = 2, 40


@dataclass
class Export:
    sessions: list[dict]
    turns: list[dict]
    labels: list[dict]
    failed: list[tuple[str, str]] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)
    outcome_known: bool = True


def index_record(record: dict) -> tuple[str, dict[str, str], dict[str, str]]:
    sid = (record.get("session") or {}).get("id") or ""
    trace_session: dict[str, str] = {}
    span_trace: dict[str, str] = {}
    for entry in record.get("traces") or []:
        tid = (entry.get("trace") or {}).get("trace_id")
        if not tid:
            continue
        trace_session[tid] = sid
        for span in entry.get("spans") or []:
            if span.get("span_id"):
                span_trace[span["span_id"]] = tid
    return sid, trace_session, span_trace


def label_rows(
    attachments: dict[str, list[dict]],
    trace_session: dict[str, str],
    span_trace: dict[str, str],
    evidence: dict[tuple[str, str], str] | None = None,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    unmapped: list[dict] = []
    for label, atts in attachments.items():
        for a in atts:
            ptype, pid = a["primitive_type"], a["primitive_id"]
            sid = tid = spid = None
            if ptype == "session":
                sid = pid
            elif ptype == "trace":
                tid = pid
                sid = trace_session.get(pid)
            elif ptype == "span":
                spid = pid
                tid = span_trace.get(pid)
                sid = trace_session.get(tid) if tid else None
            else:
                continue  # skills and other primitives are not session data
            row = {
                "label": label,
                "primitive_type": ptype,
                "primitive_id": pid,
                "session_id": sid,
                "turn_id": tid,
                "span_id": spid,
                "evidence": (evidence or {}).get((label, tid)) if tid else None,
            }
            rows.append(row)
            if sid is None:
                unmapped.append(row)
    return rows, unmapped


def session_row(sess: Session, no_outcome: set[str] | None) -> dict:
    return {
        "session_id": sess.id,
        "title": sess.title,
        "harness": sess.harness,
        "model": sess.model,
        "started_at": sess.started_at,
        "status": sess.status,
        "turn_count": sess.turn_count,
        "cost_usd": sess.cost_usd,
        "has_outcome": None if no_outcome is None else sess.id not in no_outcome,
    }


def turn_rows(sess: Session) -> list[dict]:
    return [
        {
            "session_id": sess.id,
            "turn_id": t.id,
            "ordinal": t.index,
            "user_prompt": t.user_prompt,
            "agent_text": "\n\n".join(t.assistant_text) or t.response_preview,
            "synthetic": t.synthetic,
        }
        for t in sess.turns
    ]


def _turns(item: dict) -> int:
    return int((item.get("rollup") or {}).get("turn_count") or 0)


def choose_sessions(
    labeled: dict[str, dict], recent: list[dict], sample: int, max_turns: int
) -> tuple[list[str], list[tuple[str, str]]]:
    """Every labeled session that is not too long, plus a sample of short recent ones."""
    ids: list[str] = []
    skipped: list[tuple[str, str]] = []
    for sid, it in labeled.items():
        if _turns(it) > max_turns:
            skipped.append((sid, f"{_turns(it)} turns > max {max_turns}"))
        else:
            ids.append(sid)
    extra = [
        it["id"]
        for it in recent
        if it["id"] not in labeled and RECENT_MIN_TURNS <= _turns(it) <= RECENT_MAX_TURNS
    ]
    return ids + extra[:sample], skipped


def run_export(
    paper, autolabel, cfg: Config, with_evidence: bool = False, log: Callable[[str], None] = print
) -> Export:
    labels = [lab for lab in paper.labels() if any((lab.get("usage") or {}).get(t) for t in LEVELS)]
    attachments = {
        lab["name"]: [
            a for t in LEVELS if (lab.get("usage") or {}).get(t) for a in paper.attachments(lab["id"], t)
        ]
        for lab in labels
    }
    labeled: dict[str, dict] = {}
    for lab in labels:
        if (lab.get("usage") or {}).get("session"):
            for it in paper.sessions(label=lab["name"], limit=10_000):
                labeled[it["id"]] = it
    since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent = paper.sessions(since=since, limit=cfg.sample_sessions * 3)
    ids, failed = choose_sessions(labeled, recent, cfg.sample_sessions, cfg.max_turns)

    parsed: dict[str, Session] = {}
    trace_session: dict[str, str] = {}
    span_trace: dict[str, str] = {}
    for i, sid in enumerate(ids, 1):
        log(f"[{i}/{len(ids)}] export {sid}")
        try:
            rec = paper.export_session(sid)
        except PaperError as e:
            failed.append((sid, str(e)))
            continue
        if not rec:
            continue
        _, ts, st = index_record(rec)
        trace_session.update(ts)
        span_trace.update(st)
        sess = parse_session(rec)
        if sess is not None:
            parsed[sess.id] = sess

    no_outcome: set[str] | None = None
    try:
        no_outcome = autolabel.matched_sessions(NO_OUTCOME, list(parsed))
    except AutolabelError as e:
        log(f"warning: {e}. Outcome unknown: only `golden` sessions can be training data.")

    evidence: dict[tuple[str, str], str] = {}
    if with_evidence:
        for label in CORRECTION_LABELS:
            try:
                found, reason = autolabel.turn_evidence(label, list(parsed))
            except AutolabelError as e:
                log(f"warning: {label} evidence skipped: {e}")
                continue
            if reason:
                log(f"warning: {label}: {reason}")
            evidence.update({(label, tid): text for tid, text in found.items()})

    rows, unmapped = label_rows(attachments, trace_session, span_trace, evidence)
    return Export(
        sessions=[session_row(s, no_outcome) for s in parsed.values()],
        turns=[r for s in parsed.values() for r in turn_rows(s)],
        labels=rows,
        failed=failed,
        unmapped=unmapped,
        outcome_known=no_outcome is not None,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_export(ex: Export, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_dir / "sessions.jsonl", ex.sessions)
    _write_jsonl(data_dir / "turns.jsonl", ex.turns)
    _write_jsonl(data_dir / "labels.jsonl", ex.labels)
    report = {
        "sessions": len(ex.sessions),
        "turns": len(ex.turns),
        "labels": len(ex.labels),
        "failed": ex.failed,
        "unmapped": len(ex.unmapped),
        "unmapped_by_label": _count(r["label"] for r in ex.unmapped),
        "outcome_known": ex.outcome_known,
    }
    (data_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_export.py -v`
Expected: 8 passed

- [ ] **Step 5: Add the `export` command**

In `src/tapes_eject/cli.py`, add `from .export import run_export, write_export` to the imports, and above `build_parser`:
```python
def cmd_export(cfg: config.Config, args: argparse.Namespace) -> int:
    paper = Paper(org_slug=cfg.org_slug)
    ex = run_export(paper, Autolabel(cfg.autolabel_url), cfg, with_evidence=args.evidence)
    write_export(ex, cfg.data_dir)
    print(
        f"{len(ex.sessions)} sessions, {len(ex.turns)} turns, {len(ex.labels)} labels "
        f"-> {cfg.data_dir}/ ({len(ex.failed)} failed/skipped, {len(ex.unmapped)} unmapped labels; "
        f"see report.json)"
    )
    return 0
```

In `build_parser`, before `return p`:
```python
    exp = sub.add_parser("export", help="pull labeled sessions from Paper into data/")
    exp.add_argument("--evidence", action="store_true", help="ask the cassette for per-turn evidence")
    exp.set_defaults(fn=cmd_export)
```

- [ ] **Step 6: Run a real export**

Run: `uv run tapes-eject export`
Expected: one `[i/N] export <id>` line per session, then a summary. Open `data/report.json`. The `failed` entries should be long sessions or timeouts, and `unmapped` should be small relative to `labels`. Spot-check `data/turns.jsonl` for anything secret-shaped: `grep -E 'sk-ant-|ghp_|AKIA' data/turns.jsonl` should print nothing.

- [ ] **Step 7: Commit**

```bash
git add src/tapes_eject/export.py src/tapes_eject/cli.py tests/test_export.py
git commit -m "Export labeled sessions from Paper as redacted rows"
```

---

### Task 5: Curate training examples and evaluation cases, and count them

**Files:**
- Create: `src/tapes_eject/curate.py`, `tests/test_curate.py`
- Modify: `src/tapes_eject/cli.py`

**Interfaces:**
- Consumes: the row schemas from Task 4 and `config` label constants.
- Produces: `curate.read_jsonl(path: Path) -> list[dict]`; `curate.training_examples(sessions, turns, labels) -> list[dict]`, each `{"session_id": str, "messages": [{"role": "user" | "assistant", "content": str}, ...]}` ending on an assistant message; `curate.correction_cases(turns, labels) -> list[dict]` and `curate.session_cases(sessions, turns, labels) -> list[dict]`, each `{"inputs": {"messages": [...]}, "expectations": {"guidelines": [str]}}`; `curate.eval_records(sessions, turns, labels) -> list[dict]` (both kinds combined); `curate.counts(sessions, turns, labels) -> dict[str, int]`. CLI: `tapes-eject count`.

- [ ] **Step 1: Write the failing tests**

`tests/test_curate.py`:
```python
from tapes_eject.curate import correction_cases, counts, session_cases, training_examples


def s(sid, has_outcome=True, title="t"):
    return {"session_id": sid, "has_outcome": has_outcome, "title": title}


def t(sid, tid, ordinal, prompt, reply, synthetic=False):
    return {"session_id": sid, "turn_id": tid, "ordinal": ordinal, "user_prompt": prompt, "agent_text": reply, "synthetic": synthetic}


def lab(label, sid, ptype="session", tid=None):
    return {"label": label, "primitive_type": ptype, "session_id": sid, "turn_id": tid}


TURNS = [
    t("a", "a1", 0, "add a flag", "added --x"),
    t("a", "a2", 1, "no, call it --y", "renamed to --y"),
    t("b", "b1", 0, "write tests", "wrote tests"),
    t("c", "c1", 0, "ship it", "shipped"),
]


def test_training_takes_outcome_sessions_without_negative_labels():
    ex = training_examples([s("a"), s("b"), s("c", has_outcome=False)], TURNS, [lab("pushback", "a")])
    assert [e["session_id"] for e in ex] == ["b"]
    assert ex[0]["messages"] == [
        {"role": "user", "content": "write tests"},
        {"role": "assistant", "content": "wrote tests"},
    ]


def test_golden_is_included_without_outcome_but_regression_always_excludes():
    labels = [lab("golden", "c"), lab("golden", "b"), lab("regression", "b")]
    ex = training_examples([s("b"), s("c", has_outcome=False)], TURNS, labels)
    assert [e["session_id"] for e in ex] == ["c"]


def test_unknown_outcome_is_not_training_eligible():
    assert training_examples([s("b", has_outcome=None)], TURNS, []) == []


def test_correction_case_is_the_context_before_the_corrected_turn():
    (case,) = correction_cases(TURNS, [lab("pushback", "a", "trace", "a2")])
    assert case["inputs"]["messages"] == [{"role": "user", "content": "add a flag"}]
    (g,) = case["expectations"]["guidelines"]
    assert "no, call it --y" in g


def test_first_turn_corrections_are_skipped_and_duplicates_collapse():
    labels = [
        lab("pushback", "b", "trace", "b1"),
        lab("pushback", "a", "trace", "a2"),
        lab("pushback", "a", "span", "a2"),
        lab("observation", "a", "trace", "a2"),
        lab("pushback", None, "trace", "gone"),
    ]
    assert len(correction_cases(TURNS, labels)) == 1


def test_regression_session_case_quotes_its_corrections():
    labels = [lab("regression", "a"), lab("pushback", "a", "trace", "a2")]
    (case,) = session_cases([s("a")], TURNS, labels)
    assert case["inputs"]["messages"] == [{"role": "user", "content": "add a flag"}]
    assert "no, call it --y" in case["expectations"]["guidelines"][0]


def test_golden_session_case_quotes_the_good_answer():
    (case,) = session_cases([s("b")], TURNS, [lab("golden", "b")])
    assert "wrote tests" in case["expectations"]["guidelines"][0]


def test_counts():
    c = counts([s("a"), s("b"), s("c", has_outcome=None)], TURNS, [lab("pushback", "a", "trace", "a2"), lab("golden", "b")])
    assert c == {
        "sessions": 3, "with_outcome": 2, "outcome_unknown": 1, "training_examples": 1,
        "correction_cases": 1, "golden": 1, "regression": 0, "eval_cases": 2,
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_curate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tapes_eject.curate'`

- [ ] **Step 3: Implement `curate.py`**

`src/tapes_eject/curate.py`:
```python
"""Labels decide the data. Pure functions over the rows `export` wrote."""

from __future__ import annotations

import json
from pathlib import Path

from .config import CORRECTION_LABELS, GOLDEN, NEGATIVE_LABELS, REGRESSION

MAX_CHARS = 6_000  # per message
MAX_TRAIN_TURNS = 12
MAX_CONTEXT_TURNS = 6


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def clip(text: str, n: int = MAX_CHARS) -> str:
    return text if len(text) <= n else text[:n] + "\n[...]"


def labels_by_session(labels: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in labels:
        if row.get("session_id"):
            out.setdefault(row["session_id"], set()).add(row["label"])
    return out


def turns_by_session(turns: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in sorted(turns, key=lambda r: (r["session_id"], r["ordinal"])):
        if not row.get("synthetic"):
            out.setdefault(row["session_id"], []).append(row)
    return out


def conversation(turns: list[dict]) -> list[dict]:
    msgs: list[dict] = []
    for row in turns:
        if row.get("user_prompt"):
            msgs.append({"role": "user", "content": clip(row["user_prompt"])})
        if row.get("agent_text"):
            msgs.append({"role": "assistant", "content": clip(row["agent_text"])})
    return msgs


def _trainable(labels: set[str], has_outcome: bool | None) -> bool:
    if REGRESSION in labels:
        return False  # a known failure never becomes training data, even if also golden
    return GOLDEN in labels or (has_outcome is True and not labels & NEGATIVE_LABELS)


def training_examples(sessions: list[dict], turns: list[dict], labels: list[dict]) -> list[dict]:
    by_label, by_turns = labels_by_session(labels), turns_by_session(turns)
    out: list[dict] = []
    for s in sessions:
        sid = s["session_id"]
        if not _trainable(by_label.get(sid, set()), s.get("has_outcome")):
            continue
        msgs = conversation(by_turns.get(sid, [])[:MAX_TRAIN_TURNS])
        while msgs and msgs[-1]["role"] != "assistant":
            msgs.pop()
        if any(m["role"] == "user" for m in msgs) and msgs:
            out.append({"session_id": sid, "messages": msgs})
    return out


def _correction_turns(labels: list[dict]) -> set[tuple[str, str]]:
    return {
        (r["session_id"], r["turn_id"])
        for r in labels
        if r["label"] in CORRECTION_LABELS and r.get("session_id") and r.get("turn_id")
    }


def correction_cases(turns: list[dict], labels: list[dict]) -> list[dict]:
    """The model sees the conversation up to the turn the engineer corrected, and is judged on
    whether its answer already does what the engineer later had to ask for."""
    by_turns = turns_by_session(turns)
    out: list[dict] = []
    for sid, tid in sorted(_correction_turns(labels)):
        ts = by_turns.get(sid, [])
        pos = next((i for i, row in enumerate(ts) if row["turn_id"] == tid), None)
        if not pos:  # not exported, or the first turn: no earlier agent turn to judge
            continue
        msgs = conversation(ts[max(0, pos - MAX_CONTEXT_TURNS) : pos])
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]  # the model writes that turn itself
        if not msgs:
            continue
        correction = clip(ts[pos]["user_prompt"], 2_000)
        out.append(
            {
                "inputs": {"messages": msgs},
                "expectations": {
                    "guidelines": [
                        "The response already does what the engineer later had to ask for, "
                        f"without being told: {correction}"
                    ]
                },
            }
        )
    return out


def session_cases(sessions: list[dict], turns: list[dict], labels: list[dict]) -> list[dict]:
    by_label, by_turns = labels_by_session(labels), turns_by_session(turns)
    corrections = _correction_turns(labels)
    out: list[dict] = []
    for s in sessions:
        sid = s["session_id"]
        names = by_label.get(sid, set())
        ts = by_turns.get(sid, [])
        if not ts or not ts[0].get("user_prompt") or not names & {GOLDEN, REGRESSION}:
            continue
        request = [{"role": "user", "content": clip(ts[0]["user_prompt"])}]
        if REGRESSION in names:
            said = [clip(r["user_prompt"], 500) for r in ts if (sid, r["turn_id"]) in corrections]
            failure = " | ".join(said[:3]) or s.get("title") or "the earlier failed attempt"
            guideline = (
                "The response must not repeat the failure this request once led to. "
                f"The engineer had to say: {failure}"
            )
        elif ts[0].get("agent_text"):
            guideline = (
                "The response takes the same approach as this known-good answer: "
                f"{clip(ts[0]['agent_text'], 2_000)}"
            )
        else:
            continue
        out.append({"inputs": {"messages": request}, "expectations": {"guidelines": [guideline]}})
    return out


def eval_records(sessions: list[dict], turns: list[dict], labels: list[dict]) -> list[dict]:
    return correction_cases(turns, labels) + session_cases(sessions, turns, labels)


def counts(sessions: list[dict], turns: list[dict], labels: list[dict]) -> dict[str, int]:
    by_label = labels_by_session(labels)
    corr = len(correction_cases(turns, labels))
    sess_cases = len(session_cases(sessions, turns, labels))
    return {
        "sessions": len(sessions),
        "with_outcome": sum(1 for s in sessions if s.get("has_outcome") is True),
        "outcome_unknown": sum(1 for s in sessions if s.get("has_outcome") is None),
        "training_examples": len(training_examples(sessions, turns, labels)),
        "correction_cases": corr,
        "golden": sum(1 for names in by_label.values() if GOLDEN in names),
        "regression": sum(1 for names in by_label.values() if REGRESSION in names),
        "eval_cases": corr + sess_cases,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_curate.py -v`
Expected: 8 passed

- [ ] **Step 5: Add the `count` command**

In `src/tapes_eject/cli.py`, add `from . import curate` to the imports, and above `build_parser`:
```python
def _load_rows(cfg: config.Config) -> tuple[list[dict], list[dict], list[dict]]:
    d = cfg.data_dir
    if not (d / "sessions.jsonl").exists():
        raise SystemExit(f"no export in {d}/: run `tapes-eject export` first")
    return (curate.read_jsonl(d / "sessions.jsonl"), curate.read_jsonl(d / "turns.jsonl"),
            curate.read_jsonl(d / "labels.jsonl"))


def cmd_count(cfg: config.Config, args: argparse.Namespace) -> int:
    for key, value in curate.counts(*_load_rows(cfg)).items():
        print(f"{key:>18}  {value}")
    return 0
```

In `build_parser`, before `return p`:
```python
    sub.add_parser("count", help="how much training and eval data the labels select").set_defaults(
        fn=cmd_count
    )
```

- [ ] **Step 6: Count the real data and record it**

Run: `uv run tapes-eject count`
Expected: a table of counts. Paste the output into the PR description, since the spec's data-volume risk is decided here. If `training_examples` is under 20, mark more good sessions `golden` in the console, then rerun `export` and `count` before Task 7.

- [ ] **Step 7: Commit**

```bash
git add src/tapes_eject/curate.py src/tapes_eject/cli.py tests/test_curate.py
git commit -m "Select training examples and eval cases from labels"
```

---

### Task 6: Sync to Unity Catalog and the MLflow evaluation dataset

**Files:**
- Create: `src/tapes_eject/databricks_io.py`, `src/tapes_eject/sync.py`, `tests/test_sync.py`
- Modify: `src/tapes_eject/cli.py`

**Interfaces:**
- Consumes: `config.Config`, `curate.training_examples`, `curate.eval_records`, `curate.read_jsonl`, and `export._write_jsonl` (reused to write `training.jsonl`).
- Produces: `databricks_io.Databricks(cfg, client=None)` with `sql(statement: str) -> list[list]` and `upload(local: Path, remote: str) -> None`; `sync.setup_statements(cfg) -> list[str]`; `sync.load_statements(cfg) -> list[str]`; `sync.UPLOADS = ("sessions.jsonl", "turns.jsonl", "labels.jsonl", "training.jsonl")`; `sync.rebuild_eval_dataset(cfg, records, datasets) -> object`; `sync.run_sync(cfg, db, datasets, log=print) -> dict[str, int]`. CLI: `tapes-eject sync`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sync.py`:
```python
import json

from tapes_eject.config import load
from tapes_eject.sync import load_statements, run_sync, setup_statements


def cfg(tmp_path):
    c = load({"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh"})
    return c.__class__(**{**c.__dict__, "data_dir": tmp_path})


def test_setup_creates_schema_volume_and_tables(tmp_path):
    sql = "\n".join(setup_statements(cfg(tmp_path)))
    assert "CREATE SCHEMA IF NOT EXISTS demo.agent_sessions" in sql
    assert "CREATE VOLUME IF NOT EXISTS demo.agent_sessions.raw" in sql
    for table in ("sessions", "turns", "labels"):
        assert f"CREATE TABLE IF NOT EXISTS demo.agent_sessions.{table}" in sql


def test_labels_merge_deletes_rows_missing_from_the_export(tmp_path):
    stmts = load_statements(cfg(tmp_path))
    (labels_merge,) = [s for s in stmts if "INTO demo.agent_sessions.labels" in s]
    assert "WHEN NOT MATCHED BY SOURCE THEN DELETE" in labels_merge
    assert "/Volumes/demo/agent_sessions/raw/labels.jsonl" in labels_merge
    others = [s for s in stmts if s is not labels_merge]
    assert len(others) == 3
    assert all("NOT MATCHED BY SOURCE" not in s for s in others)


def test_training_input_is_replaced_each_sync(tmp_path):
    (stmt,) = [s for s in load_statements(cfg(tmp_path)) if "training_input" in s]
    assert stmt.startswith("CREATE OR REPLACE TABLE demo.agent_sessions.training_input")


class FakeDb:
    def __init__(self):
        self.log = []

    def sql(self, statement):
        self.log.append(("sql", statement.split()[0]))
        return []

    def upload(self, local, remote):
        self.log.append(("upload", remote))


class FakeDataset:
    def __init__(self):
        self.records = None

    def merge_records(self, records):
        self.records = records
        return self


class FakeDatasets:
    def __init__(self):
        self.calls = []
        self.ds = FakeDataset()

    def delete_dataset(self, name):
        self.calls.append(("delete", name))
        raise RuntimeError("does not exist")  # first sync: nothing to delete

    def create_dataset(self, name):
        self.calls.append(("create", name))
        return self.ds


def test_run_sync_creates_before_upload_then_loads_and_rebuilds_eval(tmp_path):
    c = cfg(tmp_path)
    rows = {
        "sessions.jsonl": [{"session_id": "b", "has_outcome": True, "title": "t"}],
        "turns.jsonl": [{"session_id": "b", "turn_id": "b1", "ordinal": 0, "user_prompt": "hi", "agent_text": "yo", "synthetic": False}],
        "labels.jsonl": [{"label": "golden", "primitive_type": "session", "primitive_id": "b", "session_id": "b", "turn_id": None, "span_id": None, "evidence": None}],
    }
    for name, rs in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
    db, datasets = FakeDb(), FakeDatasets()
    got = run_sync(c, db, datasets, log=lambda m: None)
    kinds = [k for k, _ in db.log]
    assert kinds == ["sql"] * 5 + ["upload"] * 4 + ["sql"] * 4  # create, then upload, then load
    assert ("upload", "/Volumes/demo/agent_sessions/raw/training.jsonl") in db.log
    assert datasets.calls == [("delete", "demo.agent_sessions.eval_cases"), ("create", "demo.agent_sessions.eval_cases")]
    assert len(datasets.ds.records) == 1
    assert got == {"training_examples": 1, "eval_cases": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tapes_eject.sync'`

- [ ] **Step 3: Implement `databricks_io.py`**

`src/tapes_eject/databricks_io.py`:
```python
"""The two Databricks calls sync needs: run SQL on a warehouse, put a file in a volume."""

from __future__ import annotations

import time
from pathlib import Path

from .config import Config


class Databricks:
    def __init__(self, cfg: Config, client=None):
        if client is None:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient(profile=cfg.profile)
        self.w = client
        self.warehouse_id = cfg.warehouse_id

    def sql(self, statement: str) -> list[list]:
        from databricks.sdk.service.sql import StatementState

        r = self.w.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id, statement=statement, wait_timeout="50s"
        )
        while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(2)
            r = self.w.statement_execution.get_statement(r.statement_id)
        if r.status.state != StatementState.SUCCEEDED:
            msg = r.status.error.message if r.status.error else ""
            raise RuntimeError(f"{r.status.state}: {msg}\n{statement[:400]}")
        return (r.result.data_array if r.result else None) or []

    def upload(self, local: Path, remote: str) -> None:
        with open(local, "rb") as fh:
            self.w.files.upload(remote, fh, overwrite=True)
```

- [ ] **Step 4: Implement `sync.py`**

`src/tapes_eject/sync.py`:
```python
"""data/*.jsonl -> Unity Catalog tables and the MLflow evaluation dataset.

Order matters: the schema and volume must exist before files are uploaded into the volume, and
the files must be there before the MERGEs read them. Labels are the only table that deletes:
a label removed in Paper disappears here on the next sync.
"""

from __future__ import annotations

from typing import Callable

from . import curate
from .config import Config
from .export import _write_jsonl

SESSIONS_COLS = (
    "session_id STRING, title STRING, harness STRING, model STRING, started_at STRING, "
    "status STRING, turn_count INT, cost_usd DOUBLE, has_outcome BOOLEAN"
)
TURNS_COLS = (
    "session_id STRING, turn_id STRING, ordinal INT, user_prompt STRING, agent_text STRING, "
    "synthetic BOOLEAN"
)
LABELS_COLS = (
    "label STRING, primitive_type STRING, primitive_id STRING, session_id STRING, "
    "turn_id STRING, span_id STRING, evidence STRING"
)
TRAINING_COLS = "session_id STRING, messages ARRAY<STRUCT<role: STRING, content: STRING>>"
UPLOADS = ("sessions.jsonl", "turns.jsonl", "labels.jsonl", "training.jsonl")


def setup_statements(cfg: Config) -> list[str]:
    schema = f"{cfg.catalog}.{cfg.schema}"
    return [
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"CREATE VOLUME IF NOT EXISTS {schema}.{cfg.volume}",
        f"CREATE TABLE IF NOT EXISTS {cfg.table('sessions')} ({SESSIONS_COLS}, synced_at TIMESTAMP)",
        f"CREATE TABLE IF NOT EXISTS {cfg.table('turns')} ({TURNS_COLS}, synced_at TIMESTAMP)",
        f"CREATE TABLE IF NOT EXISTS {cfg.table('labels')} ({LABELS_COLS}, synced_at TIMESTAMP)",
    ]


def _source(cfg: Config, file: str, cols: str) -> str:
    return (
        f"(SELECT *, current_timestamp() AS synced_at FROM read_files("
        f"'{cfg.volume_path}/{file}', format => 'json', schema => '{cols}'))"
    )


def load_statements(cfg: Config) -> list[str]:
    return [
        f"MERGE INTO {cfg.table('sessions')} AS t USING {_source(cfg, 'sessions.jsonl', SESSIONS_COLS)} AS s "
        "ON t.session_id = s.session_id "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
        f"MERGE INTO {cfg.table('turns')} AS t USING {_source(cfg, 'turns.jsonl', TURNS_COLS)} AS s "
        "ON t.session_id = s.session_id AND t.turn_id = s.turn_id "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
        f"MERGE INTO {cfg.table('labels')} AS t USING {_source(cfg, 'labels.jsonl', LABELS_COLS)} AS s "
        "ON t.label = s.label AND t.primitive_type = s.primitive_type "
        "AND t.primitive_id = s.primitive_id "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT * "
        "WHEN NOT MATCHED BY SOURCE THEN DELETE",
        f"CREATE OR REPLACE TABLE {cfg.table('training_input')} AS SELECT * FROM read_files("
        f"'{cfg.volume_path}/training.jsonl', format => 'json', schema => '{TRAINING_COLS}')",
    ]


def rebuild_eval_dataset(cfg: Config, records: list[dict], datasets):
    """Delete and recreate, so a label removed in Paper leaves the set. MLflow keeps history."""
    name = cfg.table("eval_cases")
    try:
        datasets.delete_dataset(name=name)
    except Exception:  # first sync: there is nothing to delete; create_dataset surfaces real errors
        pass
    return datasets.create_dataset(name=name).merge_records(records)


def run_sync(cfg: Config, db, datasets, log: Callable[[str], None] = print) -> dict[str, int]:
    d = cfg.data_dir
    sessions, turns, labels = (curate.read_jsonl(d / f) for f in UPLOADS[:3])
    training = curate.training_examples(sessions, turns, labels)
    records = curate.eval_records(sessions, turns, labels)
    _write_jsonl(d / "training.jsonl", training)
    _write_jsonl(d / "eval_cases.jsonl", records)  # for inspection; MLflow holds the real one

    for stmt in setup_statements(cfg):
        db.sql(stmt)
    for name in UPLOADS:
        log(f"upload {name} -> {cfg.volume_path}/{name}")
        db.upload(d / name, f"{cfg.volume_path}/{name}")
    for stmt in load_statements(cfg):
        log(stmt.split(" AS ")[0][:80])
        db.sql(stmt)
    log(f"rebuild MLflow evaluation dataset {cfg.table('eval_cases')} ({len(records)} cases)")
    rebuild_eval_dataset(cfg, records, datasets)
    return {"training_examples": len(training), "eval_cases": len(records)}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sync.py -v`
Expected: 4 passed

- [ ] **Step 6: Add the `sync` command**

In `src/tapes_eject/cli.py`, add these imports:
```python
from .databricks_io import Databricks
from .sync import run_sync
```

Above `build_parser`:
```python
def _mlflow(cfg: config.Config):
    import mlflow

    mlflow.set_tracking_uri(f"databricks://{cfg.profile}")
    mlflow.set_registry_uri(f"databricks-uc://{cfg.profile}")
    mlflow.set_experiment(cfg.experiment)
    return mlflow


def cmd_sync(cfg: config.Config, args: argparse.Namespace) -> int:
    mlflow = _mlflow(cfg)
    import mlflow.genai.datasets as datasets

    got = run_sync(cfg, Databricks(cfg), datasets)
    print(f"{got['training_examples']} training examples, {got['eval_cases']} eval cases in "
          f"{cfg.catalog}.{cfg.schema} (MLflow {mlflow.__version__})")
    return 0
```

In `build_parser`, before `return p`:
```python
    sub.add_parser("sync", help="load data/ into Unity Catalog and the MLflow eval dataset").set_defaults(
        fn=cmd_sync
    )
```

- [ ] **Step 7: Sync for real, and prove label removal works**

Run: `uv run tapes-eject sync`
Expected: upload lines, one line per MERGE, then the counts. In the workspace, open Catalog → your catalog → `agent_sessions`. You should see `sessions`, `turns`, `labels`, `training_input`, and `eval_cases`. On `labels`, the History tab shows one version per sync.

Then prove Review Focus item 3. In the console, remove one `pushback` label from a session. Run `uv run tapes-eject export && uv run tapes-eject sync`. Then run in the SQL editor:
`SELECT count(*) FROM <catalog>.agent_sessions.labels WHERE label = 'pushback'`
Expected: one fewer row than before. Put the label back afterwards.

- [ ] **Step 8: Commit**

```bash
git add src/tapes_eject/databricks_io.py src/tapes_eject/sync.py src/tapes_eject/cli.py tests/test_sync.py
git commit -m "Sync labeled sessions to Unity Catalog and the MLflow eval dataset"
```

---

### Task 7: Train on AI Runtime (SFT) and register the model

**Files:**
- Create: `train/sft_train.py`, `databricks.yml`, `resources/sft_train.job.yml` (generated)
- Modify: `README.md` (Act 3)

**Interfaces:**
- Consumes: `/Volumes/<catalog>/agent_sessions/raw/training.jsonl` (Task 6), with rows `{"session_id", "messages": [{"role", "content"}]}`.
- Produces: the model directory `/Volumes/<catalog>/agent_sessions/raw/models/agent_qwen3_4b`, the MLflow run `sft-*` in `/Shared/tapes-eject`, and the Unity Catalog model `<catalog>.agent_sessions.agent_qwen3_4b` (logged as a vLLM `llm/v1/chat` pyfunc, ready for Model Serving). Job parameters: `catalog`, `schema`, `max_steps` (`0` = full run of 2 epochs), `method` (`full` | `lora`), `min_examples`, `base_model`.

GPU code is verified by running it as a job. No unit tests, because nothing here runs off a GPU.

- [ ] **Step 1: Write `databricks.yml`**

```yaml
bundle:
  name: tapes-eject-databricks

include:
  - resources/*.yml

sync:
  include:
    - train/*.py

targets:
  demo:
    default: true
    workspace:
      profile: tapes-eject
```

- [ ] **Step 2: Write `train/sft_train.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # SFT: Qwen3-4B on the sessions Paper's labels selected
# MAGIC Runs on AI Runtime (serverless GPU, 1x H100, AI environment v6+). Adapted from Databricks'
# MAGIC "Full fine-tuning of Qwen3-4B" tutorial. `max_steps=5` is the smoke run; `0` trains 2 epochs.

# COMMAND ----------
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "agent_sessions")
dbutils.widgets.text("max_steps", "5")
dbutils.widgets.text("method", "full")  # full on H100, lora on A10
dbutils.widgets.text("min_examples", "20")
dbutils.widgets.text("base_model", "Qwen/Qwen3-4B")
dbutils.widgets.text("experiment", "/Shared/tapes-eject")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MAX_STEPS = int(dbutils.widgets.get("max_steps"))
METHOD = dbutils.widgets.get("method")
MIN_EXAMPLES = int(dbutils.widgets.get("min_examples"))
BASE = dbutils.widgets.get("base_model")
assert CATALOG, "set the `catalog` job parameter"
assert METHOD in ("full", "lora"), METHOD

VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
OUT = f"{VOLUME}/models/agent_qwen3_4b"
UC_MODEL = f"{CATALOG}.{SCHEMA}.agent_qwen3_4b"

# COMMAND ----------
import json

with open(f"{VOLUME}/training.jsonl", encoding="utf-8") as fh:
    examples = [{"messages": json.loads(line)["messages"]} for line in fh if line.strip()]
if len(examples) < MIN_EXAMPLES:
    raise ValueError(
        f"only {len(examples)} training examples (need {MIN_EXAMPLES}); label more sessions "
        "`golden` in Paper, then `tapes-eject export && tapes-eject sync`"
    )

from datasets import Dataset

split = Dataset.from_list(examples).train_test_split(test_size=0.1, seed=7)
print(f"{len(split['train'])} train / {len(split['test'])} eval examples")

# COMMAND ----------
import importlib.util

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("experiment"))

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
peft_config = None
if METHOD == "lora":
    from peft import LoraConfig

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

args = SFTConfig(
    output_dir="/tmp/sft",
    max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=1e-5 if METHOD == "full" else 1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    max_length=4096,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="no",
    report_to="mlflow",
    use_liger_kernel=importlib.util.find_spec("liger_kernel") is not None,
)

run_name = f"sft-{METHOD}-{MAX_STEPS or 'full'}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({"training_table": f"{CATALOG}.{SCHEMA}.training_input", "examples": len(examples), "base_model": BASE, "method": METHOD})
    trainer = SFTTrainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"], processing_class=tok, peft_config=peft_config)
    trainer.train()
    final = trainer.model.merge_and_unload() if METHOD == "lora" else trainer.model
    final.save_pretrained(OUT)
    tok.save_pretrained(OUT)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register for Model Serving
# MAGIC Logged from this GPU job on purpose: logging from CPU packages CPU dependencies and the GPU
# MAGIC serving endpoint fails to start. Pattern from Databricks' "Serve custom LLMs" docs.

# COMMAND ----------
import shutil

from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel

shutil.copytree(OUT, "agent_model", dirs_exist_ok=True)


class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


metadata = {
    "task": "llm/v1/chat",
    "entrypoint": (
        "python -u -m vllm.entrypoints.openai.api_server "
        "--model agent_model --served-model-name agent "
        "--host 0.0.0.0 --port 8080 "
        "--dtype float16 --max-model-len 8192 "
        "--gpu-memory-utilization 0.85"
    ),
}

with mlflow.start_run(run_name=f"{run_name}-register"):
    info = mlflow.pyfunc.log_model(
        name="agent_qwen3_4b",
        python_model=LLMModel(),
        artifacts={"model_dir": "agent_model"},
        metadata=metadata,
        extra_pip_requirements=["mlflow==3.12.0"],
    )
    version = mlflow.register_model(info.model_uri, UC_MODEL, env_pack="databricks_model_serving")
print(f"registered {UC_MODEL} version {version.version}")
dbutils.notebook.exit(json.dumps({"model": UC_MODEL, "version": version.version, "out": OUT}))
```

- [ ] **Step 3: Deploy the files, create the GPU job once in the UI, and bring it under the bundle**

The bundle docs publish no YAML for serverless-GPU compute, so the job's compute is chosen in the UI once, then generated into YAML.

```bash
databricks bundle deploy
databricks workspace list "/Workspace/Users/$(databricks current-user me --profile tapes-eject | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')/.bundle/tapes-eject-databricks/demo/files/train"
```
Expected: `sft_train` is listed.

In the workspace UI, go to Jobs & Pipelines → Create → Job.
- Task `sft_train`, type **Notebook**, source Workspace, path = the `sft_train` file listed above.
- Compute: **Serverless GPU**. Accelerator **1xH100**, or **A10** if Task 1 Step 1 found no H100. Environment **AI v6**.
- Job parameters: `catalog=<your catalog>`, `schema=agent_sessions`, `max_steps=5`, `method=full` (`lora` on A10), `min_examples=20`, `base_model=Qwen/Qwen3-4B`, `experiment=/Shared/tapes-eject`.
- Timeout: 2 hours. Save, and note the job id.

```bash
databricks bundle generate job --existing-job-id <JOB_ID> --key sft_train
databricks bundle deployment bind sft_train <JOB_ID> --auto-approve
databricks bundle deploy
```
Expected: `resources/sft_train.job.yml` exists. If `generate` also downloaded a copy of the notebook, edit the task's `notebook_path` in the YAML to `../train/sft_train.py`, delete the downloaded copy, and run `databricks bundle deploy` again.

- [ ] **Step 4: Smoke run (5 steps)**

Run: `databricks bundle run sft_train --params max_steps=5`
Expected: the run ends `SUCCESS` and the output shows `registered <catalog>.agent_sessions.agent_qwen3_4b version 1`. In MLflow → Experiments → `/Shared/tapes-eject`, the `sft-full-5` run shows a loss curve. If it fails with CUDA out-of-memory, rerun with `--params max_steps=5,method=lora`.

Check spend before the full run: `uv run tapes-eject spend` (Task 9 adds it). If Task 9 isn't done yet, use Account console → Usage. Multiply the smoke run's GPU cost by (full steps ÷ 5) and confirm it fits the remaining budget.

- [ ] **Step 5: Full run**

Run: `databricks bundle run sft_train --params max_steps=0`
Expected: `SUCCESS`, a new model version, and an `sft-full-full` run in MLflow.

- [ ] **Step 6: Document Act 3 in the README and commit**

Append to `README.md`:
````markdown
## Act 3: Train on Databricks GPUs

`databricks bundle run sft_train --params max_steps=5` is the smoke run; `max_steps=0` is the full run. The job reads the training examples the labels selected, fine-tunes Qwen3-4B with supervised fine-tuning on one H100, logs the run to MLflow, and registers `<catalog>.agent_sessions.agent_qwen3_4b` in Unity Catalog.
````

```bash
git add databricks.yml resources train/sft_train.py README.md
git commit -m "Fine-tune Qwen3-4B on AI Runtime and register it in Unity Catalog"
```

---

### Task 8: Score base against tuned with MLflow

**Files:**
- Create: `train/eval_models.py`, `resources/eval_models.job.yml` (generated)
- Modify: `README.md` (Act 4)

**Interfaces:**
- Consumes: the MLflow evaluation dataset `<catalog>.agent_sessions.eval_cases` (Task 6), whose records have `inputs.messages` and `expectations.guidelines`, and the model directory `/Volumes/<catalog>/agent_sessions/raw/models/agent_qwen3_4b` (Task 7).
- Produces: two MLflow evaluation runs, `eval-base` and `eval-tuned`, in `/Shared/tapes-eject`, each scored with `ExpectationsGuidelines`.

- [ ] **Step 1: Write `train/eval_models.py`**

```python
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
```

- [ ] **Step 2: Smoke run on 5 cases**

Deploy, then create the job in the UI the same way as Task 7 Step 3:
- Task `eval_models`, notebook `train/eval_models.py` from the bundle files.
- Compute: **Serverless GPU**, **A10**, **AI v6**.
- Parameters: `catalog=<your catalog>`, `schema=agent_sessions`, `base_model=Qwen/Qwen3-4B`, `experiment=/Shared/tapes-eject`, `limit=5`.
- Timeout: 2 hours.

```bash
databricks bundle deploy
# create the job in the UI as above, note <EVAL_JOB_ID>
databricks bundle generate job --existing-job-id <EVAL_JOB_ID> --key eval_models
databricks bundle deployment bind eval_models <EVAL_JOB_ID> --auto-approve
databricks bundle deploy
databricks bundle run eval_models --params limit=5
```
Expected: `SUCCESS`. In MLflow, `eval-base` and `eval-tuned` each have 5 rows with a guideline verdict and rationale. If the dataset's `to_df()` has no `inputs` column, print `cases.columns` in the job output and rename it to `inputs`/`expectations` before `evaluate`. That is the only shape `mlflow.genai.evaluate` needs.

- [ ] **Step 3: Full evaluation**

Run: `databricks bundle run eval_models --params limit=0`
Expected: `SUCCESS`. Select both runs in MLflow and choose Compare. The guideline pass rate is the Act 4 number. Write both pass rates into the PR description.

- [ ] **Step 4: Document Act 4 and commit**

Append to `README.md`:
````markdown
## Act 4: Prove it

`databricks bundle run eval_models` scores the base model and the fine-tuned one against the same eval cases. Each case is a moment an engineer had to correct an agent, or a session someone labeled `golden` or `regression`. Compare the `eval-base` and `eval-tuned` runs in MLflow.
````

```bash
git add train/eval_models.py resources README.md
git commit -m "Score base and tuned models on the label-built eval cases"
```

---

### Task 9: Serve the tuned model, and track spend

**Files:**
- Create: `src/tapes_eject/serve.py`, `tests/test_serve.py`
- Modify: `src/tapes_eject/cli.py`

**Interfaces:**
- Consumes: `config.Config`, `databricks_io.Databricks`, and the Unity Catalog model from Task 7.
- Produces: `serve.ENDPOINT = "agent-qwen3-4b"`; `serve.endpoint_config(cfg, version: int)` → `databricks.sdk.service.serving.EndpointCoreConfigInput`; `serve.spend_sql(since: str) -> str`. CLI: `tapes-eject serve --version N`, `tapes-eject unserve`, `tapes-eject ask "<prompt>"`, `tapes-eject spend [--since YYYY-MM-DD]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve.py`:
```python
from databricks.sdk.service.serving import ServingModelWorkloadType

from tapes_eject.config import load
from tapes_eject.serve import endpoint_config, spend_sql

CFG = load({"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh"})


def test_endpoint_serves_the_uc_model_on_a10_and_scales_to_zero():
    conf = endpoint_config(CFG, 3)
    assert conf.name == "agent-qwen3-4b"
    (entity,) = conf.served_entities
    assert entity.entity_name == "demo.agent_sessions.agent_qwen3_4b"
    assert entity.entity_version == "3"
    assert entity.workload_type == ServingModelWorkloadType.GPU_MEDIUM
    assert entity.scale_to_zero_enabled is True


def test_spend_sql_prices_usage_since_a_date():
    sql = spend_sql("2026-09-29")
    assert "system.billing.usage" in sql and "system.billing.list_prices" in sql
    assert "usage_date >= '2026-09-29'" in sql
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_serve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tapes_eject.serve'`

- [ ] **Step 3: Implement `serve.py`**

`src/tapes_eject/serve.py`:
```python
"""The tuned model as an HTTPS endpoint, and what the demo has cost so far."""

from __future__ import annotations

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

from .config import Config

ENDPOINT = "agent-qwen3-4b"


def endpoint_config(cfg: Config, version: int) -> EndpointCoreConfigInput:
    # A10 (GPU_MEDIUM, 24 GB) holds a 4B model in fp16 and, unlike H100, scales to zero.
    return EndpointCoreConfigInput(
        name=ENDPOINT,
        served_entities=[
            ServedEntityInput(
                entity_name=cfg.table("agent_qwen3_4b"),
                entity_version=str(version),
                workload_type=ServingModelWorkloadType.GPU_MEDIUM,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
    )


def spend_sql(since: str) -> str:
    return f"""
SELECT u.sku_name,
       ROUND(SUM(u.usage_quantity), 2) AS dbus,
       ROUND(SUM(u.usage_quantity * p.pricing.default), 2) AS usd
FROM system.billing.usage u
JOIN system.billing.list_prices p
  ON u.sku_name = p.sku_name
 AND u.usage_start_time >= p.price_start_time
 AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
WHERE u.usage_date >= '{since}'
GROUP BY u.sku_name
ORDER BY usd DESC
""".strip()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_serve.py -v`
Expected: 2 passed

- [ ] **Step 5: Add the commands**

In `src/tapes_eject/cli.py`, add `from . import serve` to the imports, and above `build_parser`:
```python
def cmd_serve(cfg: config.Config, args: argparse.Namespace) -> int:
    db = Databricks(cfg)
    print(f"creating {serve.ENDPOINT} (a cold start takes minutes)...")
    db.w.serving_endpoints.create_and_wait(name=serve.ENDPOINT, config=serve.endpoint_config(cfg, args.version))
    print(f"ready: {db.w.config.host}/ml/endpoints/{serve.ENDPOINT}")
    return 0


def cmd_unserve(cfg: config.Config, args: argparse.Namespace) -> int:
    Databricks(cfg).w.serving_endpoints.delete(serve.ENDPOINT)
    print(f"deleted {serve.ENDPOINT}")
    return 0


def cmd_ask(cfg: config.Config, args: argparse.Namespace) -> int:
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    res = Databricks(cfg).w.serving_endpoints.query(
        name=serve.ENDPOINT, messages=[ChatMessage(role=ChatMessageRole.USER, content=args.prompt)]
    )
    print(res.choices[0].message.content)
    return 0


def cmd_spend(cfg: config.Config, args: argparse.Namespace) -> int:
    try:
        rows = Databricks(cfg).sql(serve.spend_sql(args.since))
    except RuntimeError as e:
        print(f"system billing tables unavailable ({e}); use Account console -> Usage")
        return 1
    total = sum(float(r[2] or 0) for r in rows)
    for sku, dbus, usd in rows:
        print(f"{sku:<50} {dbus:>10} DBU  ${usd}")
    print(f"{'total':<50} {'':>14}  ${total:.2f} of $400")
    return 0
```

In `build_parser`, before `return p`:
```python
    sv = sub.add_parser("serve", help="create the tuned model's endpoint (A10, scales to zero)")
    sv.add_argument("--version", type=int, required=True)
    sv.set_defaults(fn=cmd_serve)
    sub.add_parser("unserve", help="delete the endpoint").set_defaults(fn=cmd_unserve)
    ask = sub.add_parser("ask", help="send one prompt to the endpoint")
    ask.add_argument("prompt")
    ask.set_defaults(fn=cmd_ask)
    sp = sub.add_parser("spend", help="Databricks spend since a date, from system billing tables")
    sp.add_argument("--since", default="2026-09-29")
    sp.set_defaults(fn=cmd_spend)
```

- [ ] **Step 6: Serve, ask, delete, and check spend**

```bash
uv run tapes-eject serve --version <latest version from Task 7>
uv run tapes-eject ask "Add a --dry-run flag to the export command"
uv run tapes-eject unserve
uv run tapes-eject spend
```
Expected: `ready: ...`, then a model answer, then `deleted agent-qwen3-4b`, then a spend table with a total under $400.

- [ ] **Step 7: Run the whole suite and commit**

Run: `uv run pytest -v && uv run ruff check src tests`
Expected: all tests pass, and ruff reports no errors.

```bash
git add src/tapes_eject/serve.py src/tapes_eject/cli.py tests/test_serve.py
git commit -m "Serve the tuned model and report Databricks spend"
```

---

### Task 10: The runbook and the README story

**Files:**
- Create: `RUNBOOK.md`
- Modify: `README.md` (Acts 1 and 2, and the status line)

**Interfaces:**
- Consumes: every CLI command and job from Tasks 1–9.
- Produces: the live demo script. Nothing else depends on it.

- [ ] **Step 1: Add Acts 1 and 2 to the README**

Replace `Status: design only. No code yet.` with `Status: runnable. See RUNBOOK.md for the live demo.`, and insert before "## Act 3":
````markdown
## Act 1: Capture and label (Paper)

Paper already labels sessions. Show it live: `uv run tapes-eject label apology` finds the label across the 25 newest sessions and writes nothing. `uv run tapes-eject label apology --apply` labels them in Paper, and the chips appear in the console. Mark a few sessions `golden` and one `regression` by hand in the console.

## Act 2: Curate (Paper -> Unity Catalog)

`uv run tapes-eject export` reads every label and the sessions they sit on from Paper. `uv run tapes-eject count` shows what the labels select. `uv run tapes-eject sync` loads the tables, rebuilds the MLflow evaluation dataset, and writes the training examples. Open Catalog → `agent_sessions` to see the tables, their History, and lineage.
````

- [ ] **Step 2: Write `RUNBOOK.md`**

````markdown
# Live demo runbook (15 minutes)

Only Paper and Databricks appear on screen. Say "a supervised fine-tune of an open-weight model", never a tool name.

## The day before

- [ ] `uv run tapes-eject doctor`: all `ok`
- [ ] The autolabel cassette is running with `TYPESAFE_API_KEY` set
- [ ] `export`, `sync`, and the full `sft_train` and `eval_models` runs are done. Note the model version and both pass rates here: base ____ / tuned ____
- [ ] `uv run tapes-eject serve --version <v>`, then `ask` once to warm it. Leave it up until the demo.
- [ ] `uv run tapes-eject spend`: total $____ of $400
- [ ] Browser tabs: Paper console sessions list; Databricks Catalog → agent_sessions; MLflow experiment /Shared/tapes-eject; the endpoint page

## Act 1: Paper labels your agent work (4 min)

1. Console: the Labels column and filter. "Every session our agents run is captured and labeled."
2. Terminal: `uv run tapes-eject label pushback`. "It found where engineers pushed back, and wrote nothing."
3. `uv run tapes-eject label pushback --apply`, then refresh the console so the chips appear.
4. Mark one session `regression` in the console, live.

## Act 2: Databricks turns labels into data (4 min)

1. `uv run tapes-eject export && uv run tapes-eject count`. Read out the counts. They are real numbers.
2. `uv run tapes-eject sync`
3. Catalog → `labels` → History: a new version from this sync. Lineage: labels → training_input.
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
````

- [ ] **Step 3: Rehearse and commit**

Walk through `RUNBOOK.md` once end to end with a timer, and fill in the blanks from real runs.
Expected: under 15 minutes, and every number on screen came from a command in this repo.

```bash
git add RUNBOOK.md README.md
git commit -m "Add the live demo runbook and the four acts to the README"
```
