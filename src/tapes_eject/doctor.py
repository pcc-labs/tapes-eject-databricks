"""`tapes-eject doctor`: is this machine ready to run the demo?"""

from __future__ import annotations

import functools
import subprocess
from typing import Callable

from .config import Config

Check = tuple[str, Callable[[], str]]


def run_checks(checks: list[Check]) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for name, fn in checks:
        try:
            lines.append(f"ok    {name}: {fn()}")
        except Exception as e:  # a check's failure is its message, whatever raised it
            ok = False
            lines.append(f"FAIL  {name}: {e}")
    return ok, lines


def _paperctl(*argv: str) -> str:
    proc = subprocess.run(["paperctl", *argv], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"paperctl {argv[0]} failed")
    return proc.stdout


def _paperd() -> str:
    out = _paperctl("status")
    if "healthy" not in out:
        raise RuntimeError("paperd auth is not healthy: run `paperctl login`, then `paperctl init`")
    return "running, auth healthy"


def _paper_org() -> str:
    for line in _paperctl("whoami").splitlines():
        if line.startswith("org_slug:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("`paperctl whoami` printed no org_slug")


@functools.lru_cache(maxsize=None)
def _workspace(profile: str):
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile=profile)


def _mlflow_datasets() -> str:
    import mlflow
    import mlflow.genai.datasets as ds

    for name in ("create_dataset", "get_dataset", "delete_dataset"):
        if not hasattr(ds, name):
            raise RuntimeError(
                f"mlflow {mlflow.__version__} has no genai.datasets.{name}; need >=3.12"
            )
    return f"mlflow {mlflow.__version__}"


def checks(cfg: Config) -> list[Check]:
    w = lambda: _workspace(cfg.profile)  # noqa: E731
    return [
        ("paperd", _paperd),
        ("paper org", lambda: cfg.org_slug or _paper_org()),
        ("databricks auth", lambda: w().current_user.me().user_name),
        ("sql warehouse", lambda: str(w().warehouses.get(cfg.warehouse_id).state)),
        ("catalog", lambda: w().catalogs.get(cfg.catalog).name),
        ("mlflow eval datasets", _mlflow_datasets),
    ]
