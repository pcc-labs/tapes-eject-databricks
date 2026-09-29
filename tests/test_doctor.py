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
