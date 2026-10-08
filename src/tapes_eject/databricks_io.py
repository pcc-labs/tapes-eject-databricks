"""The two Databricks calls sync needs: run SQL on a warehouse, put a file in a volume."""

from __future__ import annotations

import time
from pathlib import Path

from .config import Config


class Databricks:
    def __init__(self, cfg: Config, client=None):
        if client is None:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient(profile=cfg.profile)
        self.w = client
        self.warehouse_id = cfg.warehouse_id

    def sql(self, statement: str) -> list[list]:
        from databricks.sdk.service.sql import StatementState

        r = self.w.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id, statement=statement, wait_timeout="50s"
        )
        while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(2)
            r = self.w.statement_execution.get_statement(r.statement_id)
        if r.status.state != StatementState.SUCCEEDED:
            msg = r.status.error.message if r.status.error else ""
            raise RuntimeError(f"{r.status.state}: {msg}\n{statement[:400]}")
        return (r.result.data_array if r.result else None) or []

    def upload(self, local: Path, remote: str) -> None:
        with open(local, "rb") as fh:
            self.w.files.upload(remote, fh, overwrite=True)
