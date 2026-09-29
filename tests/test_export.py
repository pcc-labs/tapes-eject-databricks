import json

from label_sampler.session import parse_session

from tapes_eject.autolabel import AutolabelError
from tapes_eject.config import load
from tapes_eject.export import (
    choose_sessions,
    label_rows,
    problems,
    run_export,
    scrub,
    turn_rows,
    write_export,
)
from tapes_eject.paper import PaperError
from tests.helpers import record

CFG = load(
    {
        "TAPES_EJECT_CATALOG": "demo",
        "DATABRICKS_WAREHOUSE_ID": "wh",
        "TAPES_EJECT_SAMPLE_SESSIONS": "5",
    }
)


class FakePaper:
    def __init__(self, labels, attachments, by_label, recent, records, fail=()):
        self._labels, self._att, self._by_label = labels, attachments, by_label
        self._recent, self._records, self._fail = recent, records, set(fail)
        self.exported = []

    def labels(self):
        return self._labels

    def attachments(self, label_id, primitive_type):
        return self._att.get((label_id, primitive_type), [])

    def sessions(self, label=None, since=None, limit=200):
        return self._by_label.get(label, []) if label else self._recent

    def export_session(self, session_id, timeout=300):
        self.exported.append(session_id)
        if session_id in self._fail:
            raise PaperError("paperctl sessions export timed out after 300s")
        return self._records.get(session_id)


class FakeAutolabel:
    def __init__(self, no_outcome=(), error=None, unknown=()):
        self.no_outcome, self.error, self.unknown = set(no_outcome), error, set(unknown)

    def matched_sessions(self, label, ids, chunk=25):
        if self.error:
            raise self.error
        return self.no_outcome & set(ids), self.unknown & set(ids)

    def turn_evidence(self, label, ids, chunk=25):
        return {}, None


def item(sid, turns=3, seen="2026-09-01T00:00:00Z"):
    return {"id": sid, "last_seen_at": seen, "rollup": {"turn_count": turns}}


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
    sess = parse_session(
        record("s1", [("trc_1", "use key sk-ant-abcdefghijklmnopqrstuvwxyz0123", "ok")])
    )
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
    assert report["unmapped"] == 1 and report["outcome_unknown"] == 0
    assert len((tmp_path / "labels.jsonl").read_text().splitlines()) == 3


def test_run_export_marks_outcome_unknown_when_the_cassette_is_down():
    down = FakeAutolabel(error=AutolabelError("POST .../run: unreachable"))
    ex = run_export(_paper(), down, CFG, log=lambda m: None)
    assert ex.outcome_known is False
    assert all(s["has_outcome"] is None for s in ex.sessions)


def test_span_ids_in_papers_trace_tilde_span_shape_map_through_their_trace():
    rows, unmapped = label_rows(
        {"missing-knowledge": [att("span", "trc_1~llm_9")]}, {"trc_1": "s1"}, {}
    )
    assert unmapped == []
    assert (rows[0]["session_id"], rows[0]["turn_id"], rows[0]["span_id"]) == (
        "s1",
        "trc_1",
        "trc_1~llm_9",
    )


def test_oversized_sessions_are_skipped_not_failed(tmp_path):
    paper = _paper()
    paper._by_label["pushback"] = [item("s1", turns=500)]
    ex = run_export(paper, FakeAutolabel(), CFG, log=lambda m: None)
    assert ex.skipped == [("s1", "500 turns > max 150")]
    assert ex.failed == []


def test_problems_flag_failed_exports_empty_exports_and_unknown_outcomes():
    ok = {"sessions": 3, "failed": [], "outcome_unknown": 0}
    assert problems(ok) == []
    assert problems({**ok, "failed": [["s3", "timed out"]]}) == ["1 session exports failed"]
    assert problems({**ok, "sessions": 0}) == ["no sessions exported"]
    assert problems({**ok, "outcome_unknown": 2}) == ["outcome unknown for 2 sessions"]


def test_only_the_sessions_the_cassette_could_not_read_are_outcome_unknown():
    ex = run_export(_paper(), FakeAutolabel(unknown={"s2"}), CFG, log=lambda m: None)
    assert {s["session_id"]: s["has_outcome"] for s in ex.sessions} == {"s1": True, "s2": None}


def test_papers_own_no_outcome_label_counts_as_no_outcome():
    paper = _paper()
    paper._labels.append({"id": "L2", "name": "no-outcome", "usage": {"session": 1}})
    paper._att[("L2", "session")] = [att("session", "s1")]
    ex = run_export(paper, FakeAutolabel(), CFG, log=lambda m: None)
    assert {s["session_id"]: s["has_outcome"] for s in ex.sessions}["s1"] is False


def test_an_unchanged_session_is_read_from_the_cache_on_the_next_export(tmp_path):
    first = _paper()
    run_export(first, FakeAutolabel(), CFG, log=lambda m: None, cache_dir=tmp_path)
    assert sorted(first.exported) == ["s1", "s2", "s3"]
    again = _paper()
    ex = run_export(again, FakeAutolabel(), CFG, log=lambda m: None, cache_dir=tmp_path)
    assert again.exported == ["s3"]  # s3 has no record, so nothing was cached for it
    assert {s["session_id"] for s in ex.sessions} == {"s1", "s2"}


def test_a_changed_session_is_exported_again(tmp_path):
    run_export(_paper(), FakeAutolabel(), CFG, log=lambda m: None, cache_dir=tmp_path)
    moved = _paper()
    moved._recent[1] = item("s2", seen="2026-09-02T00:00:00Z")
    run_export(moved, FakeAutolabel(), CFG, log=lambda m: None, cache_dir=tmp_path)
    assert moved.exported == ["s2", "s3"]


def test_scrub_catches_secrets_label_sampler_misses():
    text = (
        '{"api_key": "abcd1234efgh5678ijkl"} sk_live_abcdefghijklmnop '
        "postgres://admin:S3cr3tPassw0rd@db.example:5432/app "
        "dapi0123456789abcdef0123456789abcdef"
    )
    out = scrub(text)
    for secret in ("abcd1234efgh5678ijkl", "sk_live_abcdef", "S3cr3tPassw0rd", "dapi0123"):
        assert secret not in out
    assert "db.example" in out


def test_turn_rows_apply_the_extra_scrub():
    sess = parse_session(
        record("s1", [("trc_1", "connect to postgres://u:hunter2hunter2@h/db", "ok")])
    )
    (row,) = turn_rows(sess)
    assert "hunter2hunter2" not in row["user_prompt"]
