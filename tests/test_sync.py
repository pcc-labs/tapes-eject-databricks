import json

import pytest

from tapes_eject.config import load
from tapes_eject.sync import load_statements, run_sync, setup_statements, view_statements


def cfg(tmp_path):
    c = load({"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh"})
    return c.__class__(**{**c.__dict__, "data_dir": tmp_path})


def test_setup_creates_schema_volume_and_tables(tmp_path):
    sql = "\n".join(setup_statements(cfg(tmp_path)))
    assert "CREATE SCHEMA IF NOT EXISTS demo.agent_sessions" in sql
    assert "CREATE VOLUME IF NOT EXISTS demo.agent_sessions.raw" in sql
    for table in ("sessions", "turns", "labels"):
        assert f"CREATE TABLE IF NOT EXISTS demo.agent_sessions.{table}" in sql


def test_views_answer_by_model_project_and_week_with_every_session_as_denominator(tmp_path):
    views = view_statements(cfg(tmp_path))
    names = [v.split(" AS ")[0].rsplit(".", 1)[-1] for v in views]
    assert names == ["labels_by_model", "labels_by_project", "labels_by_week", "corrections"]
    by_model = views[0]
    assert "COUNT(*) AS sessions" in by_model and "GROUP BY model" in by_model
    assert "COUNT(DISTINCT s.session_id) AS sessions_with_label" in by_model
    assert "h.model <=> t.model" in by_model  # a NULL model still gets a row
    assert "date_trunc('week', to_timestamp(s.started_at))" in views[2]
    assert "t.user_prompt AS correction" in views[3] and "'pushback'" in views[3]
    assert "project STRING" in "\n".join(setup_statements(cfg(tmp_path)))


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
    def __init__(self, existing=()):
        self.existing = list(existing)  # (dataset_record_id, inputs)
        self.records = None
        self.deleted = []

    def to_df(self):
        import pandas as pd

        return pd.DataFrame(
            [{"dataset_record_id": rid, "inputs": inputs} for rid, inputs in self.existing]
        )

    def delete_records(self, record_ids):
        self.deleted = list(record_ids)
        return len(record_ids)

    def merge_records(self, records):
        self.records = records
        return self


class FakeDatasets:
    def __init__(self, existing=None):
        self.calls = []
        self.ds = FakeDataset(existing or ())
        self.exists = existing is not None

    def get_dataset(self, name):
        self.calls.append(("get", name))
        if not self.exists:
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
        return self.ds

    def create_dataset(self, name):
        self.calls.append(("create", name))
        return self.ds


def turn(sid, tid, ordinal, prompt, reply):
    return {
        "session_id": sid,
        "turn_id": tid,
        "ordinal": ordinal,
        "user_prompt": prompt,
        "agent_text": reply,
        "synthetic": False,
    }


def write_export(tmp_path, report=None):
    rows = {
        "sessions.jsonl": [
            {"session_id": "a", "has_outcome": True, "title": "t"},
            {"session_id": "b", "has_outcome": True, "title": "t"},
        ],
        "turns.jsonl": [
            turn("a", "a1", 0, "add a flag", "added --x"),
            turn("a", "a2", 1, "no, call it --y", "renamed"),
            turn("b", "b1", 0, "hi", "yo"),
        ],
        "labels.jsonl": [
            {
                "label": "pushback",
                "primitive_type": "trace",
                "primitive_id": "a2",
                "session_id": "a",
                "turn_id": "a2",
                "span_id": None,
                "evidence": None,
            }
        ],
    }
    for name, rs in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
    report = report or {"sessions": 2, "failed": [], "outcome_unknown": 0}
    (tmp_path / "report.json").write_text(json.dumps(report))


def test_run_sync_creates_before_upload_then_loads_and_builds_eval(tmp_path):
    write_export(tmp_path)
    db, datasets = FakeDb(), FakeDatasets()
    got = run_sync(cfg(tmp_path), db, datasets, log=lambda m: None)
    kinds = [k for k, _ in db.log]
    # create, then upload, then the four loads and the four views
    assert kinds == ["sql"] * 5 + ["upload"] * 4 + ["sql"] * 8
    assert ("upload", "/Volumes/demo/agent_sessions/raw/training.jsonl") in db.log
    assert datasets.calls == [
        ("get", "demo.agent_sessions.eval_cases"),
        ("create", "demo.agent_sessions.eval_cases"),
    ]
    assert len(datasets.ds.records) == 1
    assert got == {"training_examples": 1, "eval_cases": 1}


def test_resync_updates_the_dataset_in_place_and_drops_only_gone_cases(tmp_path):
    write_export(tmp_path)
    kept = {"messages": [{"role": "user", "content": "add a flag"}]}
    gone = {"messages": [{"role": "user", "content": "a label someone removed"}]}
    datasets = FakeDatasets(existing=[("r-kept", kept), ("r-gone", gone)])
    run_sync(cfg(tmp_path), FakeDb(), datasets, log=lambda m: None)
    assert ("create", "demo.agent_sessions.eval_cases") not in datasets.calls
    assert datasets.ds.deleted == ["r-gone"]
    assert datasets.ds.records[0]["inputs"] == kept


def test_sync_refuses_a_problem_export_unless_forced(tmp_path):
    write_export(tmp_path, report={"sessions": 2, "failed": [["x", "boom"]], "outcome_unknown": 0})
    db = FakeDb()
    with pytest.raises(SystemExit, match="1 session exports failed"):
        run_sync(cfg(tmp_path), db, FakeDatasets(), log=lambda m: None)
    assert db.log == []
    run_sync(cfg(tmp_path), FakeDb(), FakeDatasets(), log=lambda m: None, force=True)


def test_sync_refuses_to_empty_the_eval_dataset_unless_forced(tmp_path):
    write_export(tmp_path)
    (tmp_path / "labels.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no eval cases"):
        run_sync(cfg(tmp_path), FakeDb(), FakeDatasets(), log=lambda m: None)


def test_labels_merge_keeps_a_known_session_when_the_new_row_lost_it(tmp_path):
    (merge,) = [s for s in load_statements(cfg(tmp_path)) if "INTO demo.agent_sessions.labels" in s]
    assert "session_id = coalesce(s.session_id, t.session_id)" in merge
    assert "turn_id = coalesce(s.turn_id, t.turn_id)" in merge
