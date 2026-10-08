"""The tuned model as an HTTPS endpoint, and what the demo has cost so far."""

from __future__ import annotations

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

from .config import Config

ENDPOINT = "agent-qwen3-4b"


def endpoint_config(cfg: Config, version: int) -> EndpointCoreConfigInput:
    # A10 (GPU_MEDIUM, 24 GB) holds a 4B model in fp16 and, unlike H100, scales to zero.
    return EndpointCoreConfigInput(
        name=ENDPOINT,
        served_entities=[
            ServedEntityInput(
                entity_name=cfg.table("agent_qwen3_4b"),
                entity_version=str(version),
                workload_type=ServingModelWorkloadType.GPU_MEDIUM,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
    )


def spend_sql(since: str) -> str:
    return f"""
SELECT u.sku_name,
       ROUND(SUM(u.usage_quantity), 2) AS dbus,
       ROUND(SUM(u.usage_quantity * p.pricing.default), 2) AS usd
FROM system.billing.usage u
JOIN system.billing.list_prices p
  ON u.sku_name = p.sku_name
 AND u.usage_start_time >= p.price_start_time
 AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
WHERE u.usage_date >= '{since}'
GROUP BY u.sku_name
ORDER BY usd DESC
""".strip()
