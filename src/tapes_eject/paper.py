"""Read Paper through paperctl. paperd supplies auth and routing; this module holds no token."""

from __future__ import annotations

import json
import subprocess
from typing import Callable

Runner = Callable[[list[str], int], str]


class PaperError(RuntimeError):
    pass


def paperctl(argv: list[str], timeout: int) -> str:
    try:
        proc = subprocess.run(
            ["paperctl", *argv], capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise PaperError(f"paperctl {' '.join(argv)} timed out after {timeout}s") from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[:500]
        raise PaperError(f"paperctl {' '.join(argv)} exited {proc.returncode}: {detail}")
    return proc.stdout


class Paper:
    def __init__(self, run: Runner = paperctl, org_slug: str | None = None):
        self._run = run
        self._org = org_slug

    def _call(self, argv: list[str], timeout: int = 120) -> str:
        prefix = ["--org-slug", self._org] if self._org else []
        return self._run([*prefix, *argv], timeout)

    def labels(self) -> list[dict]:
        return json.loads(self._call(["cassettes", "labels", "list-labels"]))["labels"]

    def attachments(self, label_id: str, primitive_type: str) -> list[dict]:
        out: list[dict] = []
        cursor: str | None = None
        while True:
            argv = ["cassettes", "labels", "list-label-attachments", label_id]
            if cursor:
                argv += ["--cursor", cursor]
            argv += ["--primitive-type", primitive_type, "--limit", "200"]
            page = json.loads(self._call(argv))
            out.extend(page.get("attachments") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                return out

    def sessions(
        self, label: str | None = None, since: str | None = None, limit: int = 200
    ) -> list[dict]:
        """Session list items, newest first, paged until `limit`."""
        out: list[dict] = []
        cursor: str | None = None
        while len(out) < limit:
            argv = ["sessions", "list", "--json", "--limit", str(min(200, limit - len(out)))]
            if cursor:
                argv += ["--cursor", cursor]
            if since:
                argv += ["--since", since]
            if label:
                argv += ["--label", label]
            page = json.loads(self._call(argv))
            items = page.get("items") or []
            out.extend(items)
            cursor = page.get("next_cursor")
            if not cursor or not items:
                break
        return out[:limit]

    def export_session(self, session_id: str, timeout: int = 300) -> dict | None:
        """The full record at paperctl's default detail.

        Never pass --detail: `traces` strips spans."""
        text = self._call(["sessions", "export", session_id], timeout)
        for line in text.splitlines():
            if line.strip():
                return json.loads(line)
        return None
