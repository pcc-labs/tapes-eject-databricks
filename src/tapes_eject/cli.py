"""tapes-eject: Paper labels to Databricks datasets, evals, and a fine-tuned model."""

from __future__ import annotations

import argparse
import json
import sys

from . import config, curate, doctor, serve
from .autolabel import Autolabel
from .databricks_io import Databricks
from .export import problems, run_export, too_big, write_export
from .paper import Paper
from .sync import run_sync


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
        reason = too_big(it, cfg.max_turns, cfg.max_output_tokens)
        if reason:
            skipped.append((it["id"], reason))
        else:
            ids.append(it["id"])
    return ids, skipped


def cmd_label(cfg: config.Config, args: argparse.Namespace) -> int:
    """Find a label across the newest sessions; with --apply, label them in Paper. Sessions
    over the size caps are left out before the cassette exports anything."""
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
    paper = Paper(org_slug=cfg.org_slug)
    ex = run_export(
        paper,
        Autolabel(cfg.autolabel_url),
        cfg,
        with_evidence=args.evidence,
        cache_dir=cfg.data_dir / "cache",
    )
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
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    res = Databricks(cfg).w.serving_endpoints.query(
        name=serve.ENDPOINT, messages=[ChatMessage(role=ChatMessageRole.USER, content=args.prompt)]
    )
    print(res.choices[0].message.content)
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
    sub.add_parser("doctor", help="check paperd, the cassette, and the workspace").set_defaults(
        fn=cmd_doctor
    )
    lab = sub.add_parser("label", help="Act 1: find a label across recent sessions, then apply it")
    lab.add_argument(
        "name", help="apology, dream, subagents, no-outcome, pushback, question, observation"
    )
    lab.add_argument("--sessions", type=int, default=25)
    lab.add_argument("--apply", action="store_true", help="write the labels to Paper")
    lab.set_defaults(fn=cmd_label)
    exp = sub.add_parser("export", help="pull labeled sessions from Paper into data/")
    exp.add_argument(
        "--evidence", action="store_true", help="ask the cassette for per-turn evidence"
    )
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
    ask.set_defaults(fn=cmd_ask)
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
