"""tapes-eject: Paper labels to Databricks datasets, evals, and a fine-tuned model."""

from __future__ import annotations

import argparse
import sys

from . import config, curate, doctor
from .autolabel import Autolabel
from .databricks_io import Databricks
from .export import run_export, write_export
from .paper import Paper
from .sync import run_sync


def cmd_doctor(cfg: config.Config, args: argparse.Namespace) -> int:
    ok, lines = doctor.run_checks(doctor.checks(cfg))
    print("\n".join(lines))
    if not ok:
        print("\nFix the FAIL lines; README 'Setup' covers each one.")
    return 0 if ok else 1


def cmd_label(cfg: config.Config, args: argparse.Namespace) -> int:
    """Find a label across the newest sessions; with --apply, label them in Paper."""
    ids = [s["id"] for s in Paper(org_slug=cfg.org_slug).sessions(limit=args.sessions)]
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


def cmd_export(cfg: config.Config, args: argparse.Namespace) -> int:
    paper = Paper(org_slug=cfg.org_slug)
    ex = run_export(paper, Autolabel(cfg.autolabel_url), cfg, with_evidence=args.evidence)
    write_export(ex, cfg.data_dir)
    print(
        f"{len(ex.sessions)} sessions, {len(ex.turns)} turns, {len(ex.labels)} labels "
        f"-> {cfg.data_dir}/ ({len(ex.failed)} failed/skipped, {len(ex.unmapped)} unmapped labels; "
        f"see report.json)"
    )
    return 0


def _load_rows(cfg: config.Config) -> tuple[list[dict], list[dict], list[dict]]:
    d = cfg.data_dir
    if not (d / "sessions.jsonl").exists():
        raise SystemExit(f"no export in {d}/: run `tapes-eject export` first")
    return (curate.read_jsonl(d / "sessions.jsonl"), curate.read_jsonl(d / "turns.jsonl"),
            curate.read_jsonl(d / "labels.jsonl"))


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

    got = run_sync(cfg, Databricks(cfg), datasets)
    print(f"{got['training_examples']} training examples, {got['eval_cases']} eval cases in "
          f"{cfg.catalog}.{cfg.schema} (MLflow {mlflow.__version__})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tapes-eject")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check paperd, the cassette, and the workspace").set_defaults(
        fn=cmd_doctor
    )
    lab = sub.add_parser("label", help="Act 1: find a label across recent sessions, then apply it")
    lab.add_argument("name", help="apology, dream, subagents, no-outcome, pushback, question, observation")
    lab.add_argument("--sessions", type=int, default=25)
    lab.add_argument("--apply", action="store_true", help="write the labels to Paper")
    lab.set_defaults(fn=cmd_label)
    exp = sub.add_parser("export", help="pull labeled sessions from Paper into data/")
    exp.add_argument("--evidence", action="store_true", help="ask the cassette for per-turn evidence")
    exp.set_defaults(fn=cmd_export)
    sub.add_parser("count", help="how much training and eval data the labels select").set_defaults(
        fn=cmd_count
    )
    sub.add_parser("sync", help="load data/ into Unity Catalog and the MLflow eval dataset").set_defaults(
        fn=cmd_sync
    )
    return p


def main(argv: list[str] | None = None) -> int:
    config.load_dotenv()
    args = build_parser().parse_args(argv)
    return args.fn(config.load(), args)


if __name__ == "__main__":
    sys.exit(main())
