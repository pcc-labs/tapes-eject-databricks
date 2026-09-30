import pytest

from tapes_eject.cli import export_status, label_candidates, report_sql
from tapes_eject.config import load

CFG = load({"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh"})


def _item(sid, turns, output_tokens):
    return {"id": sid, "rollup": {"turn_count": turns, "usage": {"output_tokens": output_tokens}}}


def test_label_leaves_out_oversized_sessions_before_the_cassette_exports_them():
    items = [
        _item("small", 3, 3_900),
        _item("empty", 0, 0),
        _item("whale", 3, 796_463),
        _item("long", 351, 2_744_041),
        _item("ok", 28, 294_931),
    ]
    ids, skipped = label_candidates(items, CFG)
    assert ids == ["small", "ok"]  # the empty session is neither sent nor reported
    assert skipped == [
        ("whale", "796463 output tokens > max 400000"),
        ("long", "351 turns > max 150"),
    ]


def test_report_sql_reads_the_views_sync_created():
    sql = report_sql(CFG, "model", None)
    assert sql.startswith("SELECT model, label, sessions_with_label, sessions, rate ")
    assert "FROM demo.agent_sessions.labels_by_model ORDER BY label, rate DESC" in sql
    assert "WHERE label = 'pushback'" in report_sql(CFG, "project", "pushback")
    assert report_sql(CFG, "week", None).startswith(
        "SELECT week, label, sessions_with_label FROM demo.agent_sessions.labels_by_week"
    )
    with pytest.raises(ValueError, match="--by must be"):
        report_sql(CFG, "user", None)


def test_export_fails_on_problems_unless_partial_is_allowed():
    bad = {"sessions": 2, "failed": [["s3", "timed out"]], "outcome_unknown": 0}
    ok = {"sessions": 2, "failed": [], "outcome_unknown": 0}
    assert export_status(ok, allow_partial=False) == 0
    assert export_status(bad, allow_partial=False) == 1
    assert export_status(bad, allow_partial=True) == 0
