from tapes_eject.cli import export_status


def test_export_fails_on_problems_unless_partial_is_allowed():
    bad = {"sessions": 2, "failed": [["s3", "timed out"]], "outcome_unknown": 0}
    ok = {"sessions": 2, "failed": [], "outcome_unknown": 0}
    assert export_status(ok, allow_partial=False) == 0
    assert export_status(bad, allow_partial=False) == 1
    assert export_status(bad, allow_partial=True) == 0
