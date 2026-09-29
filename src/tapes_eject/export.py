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
            a
            for t in LEVELS
            if (lab.get("usage") or {}).get(t)
            for a in paper.attachments(lab["id"], t)
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
