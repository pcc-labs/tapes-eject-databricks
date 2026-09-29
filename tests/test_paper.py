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
            (
                ("list-label-attachments",),
                {"attachments": [{"primitive_id": "a"}], "next_cursor": "c1"},
            ),
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
