"""API tests. Need the Mars CSVs (scripts/build_mars_dataset.py) and models/serving."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from spacecraft_ai.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["models_trained_on"] == "m2020"
    assert "sky_crane" in r["phases"] and "drift" in r["fault_types"]


def test_missions_have_events(client):
    ms = {m["id"]: m for m in client.get("/api/missions").json()}
    assert set(ms) == {"msl", "m2020"}
    td = [e for e in ms["msl"]["events"] if e["event"] == "touchdown"][0]
    assert td["t_s"] == 0 and td["altitude_km"] < 0.05
    assert ms["m2020"]["models_trained_on_this_landing"] and not ms["msl"]["models_trained_on_this_landing"]


def test_trajectory(client):
    r = client.get("/api/missions/msl/trajectory", params={"step": 50}).json()
    assert len(r["t_s"]) == len(r["altitude_km"]) == len(r["phase"]) > 100
    assert r["phase"][0] == "approach" and r["phase"][-1] == "sky_crane"


def test_clean_replay_matches_module3(client):
    """Clean Curiosity replay: phases are predicted well and false alarms are few."""
    r = client.post("/api/missions/msl/replay", json={"channel": "vertical_speed_m_s"}).json()
    assert r["cross_mission"] and r["summary"]["faults_injected"] == 0
    assert r["phase_accuracy"] > 0.9
    assert r["summary"]["false_alarms"] <= 3
    assert len(r["t_s"]) == len(r["value"]) == len(r["phase_pred"]) == len(r["anomaly_score"])


def test_replay_detects_obvious_fault(client):
    r = client.post("/api/missions/msl/replay", json={
        "channel": "sensed_accel_m_s2",
        "faults": [{"kind": "spike", "phase": "approach", "magnitude_frac": 0.3}]}).json()
    f = r["faults"][0]
    assert f["kind"] == "spike" and f["phase"] == "approach" and f["detected"]
    assert any(a["matches_fault"] == 0 for a in r["alarms"])


def test_replay_validation(client):
    assert client.post("/api/missions/msl/replay", json={"faults": [{"kind": "meteor", "phase": "entry"}]}).status_code == 422
    assert client.post("/api/missions/venus/replay", json={}).status_code == 422


def test_telemetry(client):
    chans = client.get("/api/telemetry/channels").json()
    assert len(chans) == 27
    r = client.get("/api/telemetry/channels/T-8")
    if r.status_code == 503:
        pytest.skip("module-1 predictions not generated locally")
    d = r.json()
    assert d["true_positives"] == 2 and len(d["value"]) == len(d["prediction"]) == len(d["index"])
    assert client.get("/api/telemetry/channels/X-99").status_code == 404


def test_summary(client):
    s = client.get("/api/summary").json()
    assert s["fault_detection"]["or_persistence_aware"]["tp"] == 176


def test_journey(client):
    r = client.get("/api/journey")
    assert r.status_code == 200
    j = r.json()
    n = len(j["ship"])
    assert n > 100 and len(j["earth"]) == len(j["mars"]) == len(j["t_utc"]) == n
    assert j["dist_earth_km"][0] < 1e5            # starts next to Earth
    assert j["dist_mars_km"][-1] < 1e5            # ends at Mars
    assert 4.5e8 < j["total_path_km"] < 5.0e8      # NASA: ~472 million km
