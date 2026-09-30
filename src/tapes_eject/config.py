"""Settings, read from the environment. The CLI loads `.env` first."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

NEGATIVE_LABELS = frozenset(
    {"pushback", "apology", "missing-knowledge", "model-error", "regression"}
)
CORRECTION_LABELS = ("pushback", "observation", "missing-knowledge")
GOLDEN = "golden"
REGRESSION = "regression"
NO_OUTCOME = "no-outcome"

DEFAULT_AUTOLABEL_URL = "http://127.0.0.1:9996/v1/cassettes/autolabel"
# Sessions whose rollup shows more output tokens than this are never exported. Turn count
# alone misses a session of three turns whose tool output runs to hundreds of megabytes, and
# exporting one of those has taken Paper's export service down.
DEFAULT_MAX_OUTPUT_TOKENS = 400_000


@dataclass(frozen=True)
class Config:
    catalog: str
    warehouse_id: str
    profile: str = "tapes-eject"
    schema: str = "agent_sessions"
    volume: str = "raw"
    autolabel_url: str = DEFAULT_AUTOLABEL_URL
    org_slug: str | None = None
    sample_sessions: int = 200
    max_turns: int = 150
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    export_pause: float = 1.0  # seconds between export requests; Paper's export dislikes bursts
    skip_sessions: frozenset[str] = frozenset()  # ids whose export takes Paper's service down
    experiment: str = "/Shared/tapes-eject"
    data_dir: Path = Path("data")

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"


def load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=value lines into the environment. Variables already set win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load(env: dict[str, str] | None = None) -> Config:
    env = dict(os.environ) if env is None else env
    missing = [k for k in ("TAPES_EJECT_CATALOG", "DATABRICKS_WAREHOUSE_ID") if not env.get(k)]
    if missing:
        raise SystemExit(f"missing settings: {', '.join(missing)} (copy .env.example to .env)")
    return Config(
        catalog=env["TAPES_EJECT_CATALOG"],
        warehouse_id=env["DATABRICKS_WAREHOUSE_ID"],
        profile=env.get("DATABRICKS_CONFIG_PROFILE") or "tapes-eject",
        autolabel_url=(env.get("AUTOLABEL_URL") or DEFAULT_AUTOLABEL_URL).rstrip("/"),
        org_slug=env.get("PAPER_ORG_SLUG") or None,
        sample_sessions=int(env.get("TAPES_EJECT_SAMPLE_SESSIONS") or 200),
        max_turns=int(env.get("TAPES_EJECT_MAX_TURNS") or 150),
        max_output_tokens=int(
            env.get("TAPES_EJECT_MAX_OUTPUT_TOKENS") or DEFAULT_MAX_OUTPUT_TOKENS
        ),
        export_pause=float(env.get("TAPES_EJECT_EXPORT_PAUSE") or 1.0),
        skip_sessions=frozenset(
            s.strip() for s in (env.get("TAPES_EJECT_SKIP_SESSIONS") or "").split(",") if s.strip()
        ),
    )


def ping_url(base: str) -> str:
    """The cassette answers /ping at its host root, whatever its route prefix."""
    parts = urlsplit(base)
    return f"{parts.scheme}://{parts.netloc}/ping"
