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


def test_replay_keeps_faults_visible(client):
    """A 0.1 s spike survives the thinning of the curve, and each fault reports how much it moved the reading."""
    r = client.post("/api/missions/msl/replay", json={
        "channel": "vertical_speed_m_s",
        "faults": [{"kind": "spike", "phase": "entry"}, {"kind": "dropout", "phase": "sky_crane"}]}).json()
    assert len(r["value_healthy"]) == len(r["value"]) == len(r["t_s"])
    assert all(b > a for a, b in zip(r["t_s"], r["t_s"][1:]))
    spike, dropout = sorted(r["faults"], key=lambda f: f["start_t"])
    gap = max(abs(v - h) for v, h in zip(r["value"], r["value_healthy"]))
    assert gap >= 0.99 * spike["max_change"] > 0        # the spike is in the curve the app draws
    assert dropout["max_change_frac"] < 0.005           # vertical speed is ~0 under the sky crane anyway


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


def test_replay_with_wind_gust(client):
    """A gust moves the spacecraft: every sensor reads it, nothing is broken, alarms during it are not false alarms."""
    r = client.post("/api/missions/msl/replay", json={
        "channel": "vertical_speed_m_s", "gusts": [{"phase": "parachute", "strength": "severe"}]}).json()
    g = r["gusts"][0]
    assert g["phase"] == "parachute" and g["peak_wind_m_s"] == 35 and g["drift_at_touchdown_m"] > 50
    assert g["max_change"] > 0.6 and r["summary"]["gusts"] == 1
    assert r["wind"] and len(r["wind"]["drift_e_m"]) == len(r["t_s"])
    assert max(abs(v - n) for v, n in zip(r["value"], r["value_nasa"])) > 0.6     # the reading changed ...
    assert r["value"] == r["value_healthy"]                                       # ... and the sensor is fine
    assert all(a["matches_gust"] is None or a["matches_fault"] is None for a in r["alarms"])
    assert r["summary"]["false_alarms"] == sum(a["matches_fault"] is None and a["matches_gust"] is None
                                               for a in r["alarms"])
    assert client.post("/api/missions/msl/replay", json={"gusts": [{"phase": "approach"}]}).status_code == 422


def test_gust_gets_no_credit_for_alarms_that_happen_anyway(client):
    """Curiosity, seed 3: the AI raises a false alarm early in entry with or without wind. A gust
    there must not count as noticed, and that alarm stays a false alarm."""
    body = {"channel": "vertical_speed_m_s", "seed": 3}
    calm = client.post("/api/missions/msl/replay", json=body).json()
    windy = client.post("/api/missions/msl/replay",
                        json={**body, "gusts": [{"phase": "entry", "strength": "severe"}]}).json()
    g = windy["gusts"][0]
    first = calm["alarms"][0]
    assert first["matches_fault"] is None                                  # a false alarm without wind
    assert g["start_t"] <= first["end_t"] and first["start_t"] <= g["effect_end_t"]   # inside the gust
    # this gust adds no alarm of its own: same alarms, nothing credited to the wind
    assert [a["start_t"] for a in windy["alarms"]] == [a["start_t"] for a in calm["alarms"]]
    assert all(a["matches_gust"] is None for a in windy["alarms"])
    assert g["noticed"] is False and windy["summary"]["gusts_noticed"] == 0
    assert windy["summary"]["false_alarms"] == calm["summary"]["false_alarms"]
