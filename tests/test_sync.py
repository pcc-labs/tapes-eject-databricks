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
        "turns.jsonl": [
            {
                "session_id": "b",
                "turn_id": "b1",
                "ordinal": 0,
                "user_prompt": "hi",
                "agent_text": "yo",
                "synthetic": False,
            }
        ],
        "labels.jsonl": [
            {
                "label": "golden",
                "primitive_type": "session",
                "primitive_id": "b",
                "session_id": "b",
                "turn_id": None,
                "span_id": None,
                "evidence": None,
            }
        ],
    }
    for name, rs in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
    db, datasets = FakeDb(), FakeDatasets()
    got = run_sync(c, db, datasets, log=lambda m: None)
    kinds = [k for k, _ in db.log]
    assert kinds == ["sql"] * 5 + ["upload"] * 4 + ["sql"] * 4  # create, then upload, then load
    assert ("upload", "/Volumes/demo/agent_sessions/raw/training.jsonl") in db.log
    assert datasets.calls == [
        ("delete", "demo.agent_sessions.eval_cases"),
        ("create", "demo.agent_sessions.eval_cases"),
    ]
    assert len(datasets.ds.records) == 1
    assert got == {"training_examples": 1, "eval_cases": 1}
