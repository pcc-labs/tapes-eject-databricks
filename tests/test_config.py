import pytest

from tapes_eject.config import load, ping_url

BASE = {"TAPES_EJECT_CATALOG": "demo", "DATABRICKS_WAREHOUSE_ID": "wh1"}


def test_load_requires_catalog_and_warehouse():
    with pytest.raises(SystemExit, match="TAPES_EJECT_CATALOG"):
        load({"DATABRICKS_WAREHOUSE_ID": "wh1"})
    with pytest.raises(SystemExit, match="DATABRICKS_WAREHOUSE_ID"):
        load({"TAPES_EJECT_CATALOG": "demo"})


def test_load_defaults_and_names():
    cfg = load({**BASE, "AUTOLABEL_URL": "http://h:9996/v1/cassettes/autolabel/"})
    assert cfg.table("labels") == "demo.agent_sessions.labels"
    assert cfg.volume_path == "/Volumes/demo/agent_sessions/raw"
    assert cfg.autolabel_url == "http://h:9996/v1/cassettes/autolabel"
    assert cfg.profile == "tapes-eject"
    assert cfg.sample_sessions == 200 and cfg.max_turns == 150
    assert cfg.max_output_tokens == 400_000
    assert load({**BASE, "TAPES_EJECT_MAX_OUTPUT_TOKENS": "50000"}).max_output_tokens == 50_000
    assert cfg.experiment == "/Shared/tapes-eject"
    assert cfg.org_slug is None


def test_ping_url_is_the_host_root():
    assert ping_url("http://127.0.0.1:9996/v1/cassettes/autolabel") == "http://127.0.0.1:9996/ping"
    assert ping_url("https://box.example/api/autolabel") == "https://box.example/ping"
