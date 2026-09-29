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
