"""data/*.jsonl -> Unity Catalog tables and the MLflow evaluation dataset.

Order matters: the schema and volume must exist before files are uploaded into the volume, and
the files must be there before the MERGEs read them. Labels are the only table that deletes:
a label removed in Paper disappears here on the next sync. An export with failures, or one that
would leave no eval cases, is refused unless forced: syncing it would wipe good data.
"""

from __future__ import annotations

import json
from typing import Callable

from . import curate
from .config import Config
from .export import _write_jsonl, problems

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
        f"CREATE TABLE IF NOT EXISTS {cfg.table('sessions')} "
        f"({SESSIONS_COLS}, synced_at TIMESTAMP)",
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
        f"MERGE INTO {cfg.table('sessions')} AS t "
        f"USING {_source(cfg, 'sessions.jsonl', SESSIONS_COLS)} AS s "
        "ON t.session_id = s.session_id "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
        f"MERGE INTO {cfg.table('turns')} AS t "
        f"USING {_source(cfg, 'turns.jsonl', TURNS_COLS)} AS s "
        "ON t.session_id = s.session_id AND t.turn_id = s.turn_id "
        "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
        f"MERGE INTO {cfg.table('labels')} AS t "
        f"USING {_source(cfg, 'labels.jsonl', LABELS_COLS)} AS s "
        "ON t.label = s.label AND t.primitive_type = s.primitive_type "
        "AND t.primitive_id = s.primitive_id "
        "WHEN MATCHED THEN UPDATE SET "
        "session_id = coalesce(s.session_id, t.session_id), "
        "turn_id = coalesce(s.turn_id, t.turn_id), "
        "span_id = coalesce(s.span_id, t.span_id), "
        "evidence = coalesce(s.evidence, t.evidence), "
        "synced_at = s.synced_at "
        "WHEN NOT MATCHED THEN INSERT * "
        "WHEN NOT MATCHED BY SOURCE THEN DELETE",
        f"CREATE OR REPLACE TABLE {cfg.table('training_input')} AS SELECT * FROM read_files("
        f"'{cfg.volume_path}/training.jsonl', format => 'json', schema => '{TRAINING_COLS}')",
    ]


def _key(inputs) -> str:
    return json.dumps(inputs, sort_keys=True)


def rebuild_eval_dataset(cfg: Config, records: list[dict], datasets):
    """Update the dataset in place so MLflow keeps its version history: drop the cases whose
    labels are gone, then merge the rest (merge_records upserts by inputs)."""
    name = cfg.table("eval_cases")
    try:
        ds = datasets.get_dataset(name=name)
    except Exception:  # first sync: no dataset yet; create_dataset surfaces real errors
        ds = datasets.create_dataset(name=name)
    want = {_key(r["inputs"]) for r in records}
    df = ds.to_df()
    if len(df):
        gone = [
            row["dataset_record_id"] for _, row in df.iterrows() if _key(row["inputs"]) not in want
        ]
        if gone:
            ds.delete_records(gone)
    return ds.merge_records(records)


def run_sync(
    cfg: Config, db, datasets, log: Callable[[str], None] = print, force: bool = False
) -> dict[str, int]:
    d = cfg.data_dir
    report_path = d / "report.json"
    if not report_path.exists():
        raise SystemExit(f"no export in {d}/: run `tapes-eject export` first")
    issues = problems(json.loads(report_path.read_text(encoding="utf-8")))
    if issues and not force:
        raise SystemExit(
            f"refusing to sync this export: {'; '.join(issues)} (--force to sync anyway)"
        )
    sessions, turns, labels = (curate.read_jsonl(d / f) for f in UPLOADS[:3])
    training = curate.training_examples(sessions, turns, labels)
    records = curate.eval_records(sessions, turns, labels)
    _write_jsonl(d / "training.jsonl", training)
    _write_jsonl(d / "eval_cases.jsonl", records)  # for inspection; MLflow holds the real one
    if not records and not force:
        raise SystemExit("refusing to sync: no eval cases, which would empty eval_cases (--force)")

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
