from databricks.sdk.service.serving import ServingModelWorkloadType

from tapes_eject.config import load
from tapes_eject.serve import endpoint_config, spend_sql

CFG = load({"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh"})


def test_endpoint_serves_the_uc_model_on_a10_and_scales_to_zero():
    conf = endpoint_config(CFG, 3)
    assert conf.name == "agent-qwen3-4b"
    (entity,) = conf.served_entities
    assert entity.entity_name == "demo.agent_sessions.agent_qwen3_4b"
    assert entity.entity_version == "3"
    assert entity.workload_type == ServingModelWorkloadType.GPU_MEDIUM
    assert entity.scale_to_zero_enabled is True


def test_spend_sql_prices_usage_since_a_date():
    sql = spend_sql("2026-09-29")
    assert "system.billing.usage" in sql and "system.billing.list_prices" in sql
    assert "usage_date >= '2026-09-29'" in sql
