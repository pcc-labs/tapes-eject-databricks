"""tapes-eject: labeled agent sessions to Databricks tables, evals, and a fine-tuned model."""

from __future__ import annotations

import argparse
import json
import sys

from . import config, curate, doctor, serve
from .autolabel import Autolabel
from .databricks_io import Databricks
from .export import _cached_export, problems, run_export, too_big, write_export
from .paper import Paper
from .session import parse_session
from .sync import run_sync
from .tapes import (
    AUTO_LABELS,
    MANUAL_LABELS,
    Tapes,
    find_labels,
    mark,
    merge_auto,
    read_labels,
    write_labels,
)


def cmd_doctor(cfg: config.Config, args: argparse.Namespace) -> int:
    ok, lines = doctor.run_checks(doctor.checks(cfg))
    print("\n".join(lines))
    if not ok:
        print("\nFix the FAIL lines; README 'Setup' covers each one.")
    return 0 if ok else 1


def label_candidates(
    items: list[dict], cfg: config.Config
) -> tuple[list[str], list[tuple[str, str]]]:
    """(ids to send to the cassette, (id, reason) for each session over the size caps)."""
    ids: list[str] = []
    skipped: list[tuple[str, str]] = []
    for it in items:
        if not ((it.get("rollup") or {}).get("turn_count") or 0):
            continue  # an empty session cannot carry a label; do not export it
        reason = too_big(it, cfg.max_turns, cfg.max_output_tokens, cfg.skip_sessions)
        if reason:
            skipped.append((it["id"], reason))
        else:
            ids.append(it["id"])
    return ids, skipped


def source(cfg: config.Config):
    """Where sessions come from: a local tapes stack (default) or a Paper org."""
    if cfg.source == "paper":
        return Paper(org_slug=cfg.org_slug)
    return Tapes(cfg.tapes_api, cfg.labels_path)


def label_local(cfg: config.Config, args: argparse.Namespace) -> int:
    """Find `pushback` and `apology` turns in the newest local sessions and keep them in
    data/local_labels.jsonl. Hand-set labels are never touched."""
    tapes = Tapes(cfg.tapes_api, cfg.labels_path)
    items = tapes.sessions(limit=args.sessions)
    ids, skipped = label_candidates(items, cfg)
    for sid, reason in skipped:
        print(f"  skip {sid}: {reason}")
    seen = {it["id"]: it.get("last_seen_at") for it in items}
    found: list[dict] = []
    scanned: set[str] = set()
    for i, sid in enumerate(ids, 1):
        print(f"  [{i}/{len(ids)}] {sid}", file=sys.stderr)
        rec, _ = _cached_export(tapes, sid, seen.get(sid), cfg.data_dir / "cache")
        sess = parse_session(rec) if rec else None
        if sess is None:
            continue
        scanned.add(sid)
        found.extend(find_labels(sess))
    write_labels(cfg.labels_path, merge_auto(read_labels(cfg.labels_path), found, scanned))
    for name in AUTO_LABELS:
        if args.name and name != args.name:
            continue
        rows = [r for r in found if r["label"] == name]
        sessions = len({r["session_id"] for r in rows})
        print(f"{name}: {len(rows)} turns in {sessions} of {len(scanned)} sessions")
        for r in rows[: args.show]:
            print(f"  {r['session_id'][:13]}  {r['evidence']}")
    print(f"labels -> {cfg.labels_path} (pattern-matched: read the evidence, fix with `mark`)")
    return 0


def cmd_mark(cfg: config.Config, args: argparse.Namespace) -> int:
    if cfg.source == "paper":
        print("with TAPES_EJECT_SOURCE=paper, set labels in the Paper console")
        return 1
    rows = mark(read_labels(cfg.labels_path), args.label, args.session_ids, args.remove)
    write_labels(cfg.labels_path, rows)
    verb = "removed from" if args.remove else "added to"
    print(f"{args.label} {verb} {len(args.session_ids)} session(s) in {cfg.labels_path}")
    return 0


def cmd_label(cfg: config.Config, args: argparse.Namespace) -> int:
    """Local tapes: find labels by pattern. Paper: find a label through the autolabel cassette,
    and with --apply write it to Paper. Sessions over the size caps are left out either way."""
    if cfg.source != "paper":
        return label_local(cfg, args)
    if not args.name:
        print("with TAPES_EJECT_SOURCE=paper, name the label to find, e.g. `label pushback`")
        return 1
    items = Paper(org_slug=cfg.org_slug).sessions(limit=args.sessions)
    ids, skipped = label_candidates(items, cfg)
    for sid, reason in skipped:
        print(f"  skip {sid}: {reason}")
    if not ids:
        print(f"{args.name}: none of the {len(items)} newest sessions are under the size caps")
        return 1
    client = Autolabel(cfg.autolabel_url)
    res = client.run(args.name, ids, apply=False, on_progress=lambda p: print(f"  {p}"))
    if res.get("reason"):
        print(f"{args.name}: {res['reason']}")
        return 1
    print(f"{args.name} found in {res['matched']} of {len(ids)} sessions")
    if not args.apply or not res["matched"]:
        return 0
    done = client.run(args.name, ids, apply=True, on_progress=lambda p: print(f"  {p}"))
    print(f"{args.name}: {done['created']} new labels, {done['pushed']} pushed to Paper")
    return 0 if not done["push_failed"] else 1


def export_status(report: dict, allow_partial: bool) -> int:
    issues = problems(report)
    for issue in issues:
        print(f"problem: {issue}")
    if issues and not allow_partial:
        print("sync will refuse this export; rerun `export`, or pass --allow-partial to accept it")
        return 1
    return 0


def cmd_export(cfg: config.Config, args: argparse.Namespace) -> int:
    ex = run_export(source(cfg), cfg, cache_dir=cfg.data_dir / "cache")
    write_export(ex, cfg.data_dir)
    print(
        f"{len(ex.sessions)} sessions, {len(ex.turns)} turns, {len(ex.labels)} labels "
        f"-> {cfg.data_dir}/ ({len(ex.failed)} failed, {len(ex.skipped)} skipped, "
        f"{len(ex.unmapped)} unmapped labels; see report.json)"
    )
    report = json.loads((cfg.data_dir / "report.json").read_text(encoding="utf-8"))
    return export_status(report, args.allow_partial)


def _load_rows(cfg: config.Config) -> tuple[list[dict], list[dict], list[dict]]:
    d = cfg.data_dir
    if not (d / "sessions.jsonl").exists():
        raise SystemExit(f"no export in {d}/: run `tapes-eject export` first")
    return (
        curate.read_jsonl(d / "sessions.jsonl"),
        curate.read_jsonl(d / "turns.jsonl"),
        curate.read_jsonl(d / "labels.jsonl"),
    )


def cmd_count(cfg: config.Config, args: argparse.Namespace) -> int:
    for key, value in curate.counts(*_load_rows(cfg)).items():
        print(f"{key:>18}  {value}")
    return 0


def _mlflow(cfg: config.Config):
    import mlflow

    mlflow.set_tracking_uri(f"databricks://{cfg.profile}")
    mlflow.set_registry_uri(f"databricks-uc://{cfg.profile}")
    mlflow.set_experiment(cfg.experiment)
    return mlflow


def cmd_sync(cfg: config.Config, args: argparse.Namespace) -> int:
    mlflow = _mlflow(cfg)
    import mlflow.genai.datasets as datasets

    got = run_sync(cfg, Databricks(cfg), datasets, force=args.force)
    print(
        f"{got['training_examples']} training examples, {got['eval_cases']} eval cases in "
        f"{cfg.catalog}.{cfg.schema} (MLflow {mlflow.__version__})"
    )
    return 0


def cmd_serve(cfg: config.Config, args: argparse.Namespace) -> int:
    db = Databricks(cfg)
    print(f"creating {serve.ENDPOINT} (a cold start takes minutes)...")
    db.w.serving_endpoints.create_and_wait(
        name=serve.ENDPOINT, config=serve.endpoint_config(cfg, args.version)
    )
    print(f"ready: {db.w.config.host}/ml/endpoints/{serve.ENDPOINT}")
    return 0


def cmd_unserve(cfg: config.Config, args: argparse.Namespace) -> int:
    Databricks(cfg).w.serving_endpoints.delete(serve.ENDPOINT)
    print(f"deleted {serve.ENDPOINT}")
    return 0


def cmd_ask(cfg: config.Config, args: argparse.Namespace) -> int:
    if args.url:  # train/local_serve.py on a local GPU instead of Model Serving
        from .local_gpu import ask_local

        print(ask_local(args.url, args.prompt))
        return 0
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    res = Databricks(cfg).w.serving_endpoints.query(
        name=serve.ENDPOINT, messages=[ChatMessage(role=ChatMessageRole.USER, content=args.prompt)]
    )
    print(res.choices[0].message.content)
    return 0


REPORT_DIMS = ("model", "project", "week")


def report_sql(cfg: config.Config, by: str, label: str | None) -> str:
    """The Act 2 question: which <by> carries each label most, from the views sync created."""
    if by not in REPORT_DIMS:
        raise ValueError(f"--by must be one of {REPORT_DIMS}, not {by!r}")
    where = f" WHERE label = '{label}'" if label else ""
    if by == "week":
        return (
            f"SELECT week, label, sessions_with_label FROM {cfg.table('labels_by_week')}"
            f"{where} ORDER BY week, label"
        )
    return (
        f"SELECT {by}, label, sessions_with_label, sessions, rate "
        f"FROM {cfg.table('labels_by_' + by)}{where} ORDER BY label, rate DESC, sessions DESC"
    )


def cmd_report(cfg: config.Config, args: argparse.Namespace) -> int:
    rows = Databricks(cfg).sql(report_sql(cfg, args.by, args.label))
    if not rows:
        print("no rows: run `tapes-eject sync` first, or the labels selected nothing")
        return 1
    if args.by == "week":
        print(f"{'week':<12} {'label':<18} {'sessions':>8}")
        for week, label, n in rows:
            print(f"{str(week)[:10]:<12} {label:<18} {n:>8}")
        return 0
    print(f"{args.by:<36} {'label':<18} {'with':>5} {'of':>5} {'rate':>6}")
    for dim, label, hits, total, rate in rows:
        print(f"{str(dim):<36} {label:<18} {hits:>5} {total:>5} {float(rate):>6.1%}")
    return 0


def cmd_spend(cfg: config.Config, args: argparse.Namespace) -> int:
    try:
        rows = Databricks(cfg).sql(serve.spend_sql(args.since))
    except RuntimeError as e:
        print(f"system billing tables unavailable ({e}); use Account console -> Usage")
        return 1
    total = sum(float(r[2] or 0) for r in rows)
    for sku, dbus, usd in rows:
        print(f"{sku:<50} {dbus:>10} DBU  ${usd}")
    print(f"{'total':<50} {'':>14}  ${total:.2f} of $400")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tapes-eject")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the session source and the workspace").set_defaults(
        fn=cmd_doctor
    )
    lab = sub.add_parser("label", help="find pushback and apology turns in recent sessions")
    lab.add_argument(
        "name", nargs="?", help="show only this label; required with TAPES_EJECT_SOURCE=paper"
    )
    lab.add_argument("--sessions", type=int, default=200, help="how many recent sessions to scan")
    lab.add_argument("--show", type=int, default=5, help="examples to print per label")
    lab.add_argument("--apply", action="store_true", help="Paper only: write the labels to Paper")
    lab.set_defaults(fn=cmd_label)
    mk = sub.add_parser("mark", help="label whole sessions by hand, e.g. golden or regression")
    mk.add_argument("label", choices=MANUAL_LABELS + AUTO_LABELS)
    mk.add_argument("session_ids", nargs="+")
    mk.add_argument("--remove", action="store_true", help="drop the label from these sessions")
    mk.set_defaults(fn=cmd_mark)
    exp = sub.add_parser("export", help="pull sessions, turns, and labels into data/")
    exp.add_argument(
        "--allow-partial", action="store_true", help="exit 0 even if some exports failed"
    )
    exp.set_defaults(fn=cmd_export)
    sub.add_parser("count", help="how much training and eval data the labels select").set_defaults(
        fn=cmd_count
    )
    sy = sub.add_parser("sync", help="load data/ into Unity Catalog and the MLflow eval dataset")
    sy.add_argument("--force", action="store_true", help="sync even a partial or empty export")
    sy.set_defaults(fn=cmd_sync)
    sv = sub.add_parser("serve", help="create the tuned model's endpoint (A10, scales to zero)")
    sv.add_argument("--version", type=int, required=True)
    sv.set_defaults(fn=cmd_serve)
    sub.add_parser("unserve", help="delete the endpoint").set_defaults(fn=cmd_unserve)
    ask = sub.add_parser("ask", help="send one prompt to the endpoint")
    ask.add_argument("prompt")
    ask.add_argument(
        "--url", help="a local train/local_serve.py instead, e.g. http://127.0.0.1:8081"
    )
    ask.set_defaults(fn=cmd_ask)
    rp = sub.add_parser("report", help="Act 2: which model, project, or week carries each label")
    rp.add_argument("--by", choices=REPORT_DIMS, default="model")
    rp.add_argument("--label", help="one label only, e.g. pushback")
    rp.set_defaults(fn=cmd_report)
    sp = sub.add_parser("spend", help="Databricks spend since a date, from system billing tables")
    sp.add_argument("--since", default="2026-09-29")
    sp.set_defaults(fn=cmd_spend)
    return p


def main(argv: list[str] | None = None) -> int:
    config.load_dotenv()
    args = build_parser().parse_args(argv)
    return args.fn(config.load(), args)


if __name__ == "__main__":
    sys.exit(main())
