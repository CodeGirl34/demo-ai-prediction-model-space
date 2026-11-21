"""Wind gust model. Needs the Mars CSVs (scripts/build_mars_dataset.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from spacecraft_ai.faults import wind

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "mars_edl"


@pytest.fixture(scope="module")
def msl():
    f = DATA / "msl_edl.csv"
    if not f.exists():
        pytest.skip("run scripts/build_mars_dataset.py first")
    return pd.read_csv(f)


def _gust(df, phase, strength="strong", seed=0):
    return wind.random_gust(df, np.random.default_rng(seed), phase, strength)


def test_frame_reproduces_nasa_sensed_acceleration(msl):
    """The local-frame physics used to move the spacecraft agrees with NASA's own numbers."""
    v, a = wind.nominal_motion(msl)
    late = msl.phase.isin(["parachute", "powered_descent", "sky_crane"]).to_numpy()
    err = np.abs(np.linalg.norm(a, axis=1) - msl.sensed_accel_m_s2.to_numpy())[late]
    assert np.median(err) < 0.05
    assert np.abs(np.linalg.norm(v, axis=1) - msl.speed_rel_mars_m_s).max() < 1.0
    assert np.abs(v[:, 2] - msl.vertical_speed_m_s).max() < 0.5


def test_no_gust_changes_nothing(msl):
    out, info = wind.simulate(msl, [])
    pd.testing.assert_frame_equal(out, msl)
    assert not info["drift_m"].any()


def test_no_gusts_without_air(msl):
    with pytest.raises(ValueError):
        _gust(msl, "approach")


def test_parachute_gust_pushes_the_lander_off_course(msl):
    g = _gust(msl, "parachute", "strong")
    out, info = wind.simulate(msl, [g])
    drift = np.linalg.norm(info["drift_m"][:, :2], axis=1)
    assert 50 < drift[-1] < 600                                   # lands where the wind left it
    assert not np.any(info["drift_m"][: g.start])                 # nothing before the gust
    # one consistent motion: altitude change = integral of the vertical speed change
    dz = (out.altitude_km - msl.altitude_km).to_numpy() * 1000
    dvz = (out.vertical_speed_m_s - msl.vertical_speed_m_s).to_numpy()
    assert np.allclose(dz, np.cumsum(dvz) * 0.1, atol=0.05)
    assert abs(dz[-1]) < 1.0                                      # guidance restored the altitude profile
    assert (out.sensed_accel_m_s2 - msl.sensed_accel_m_s2).abs().max() > 0.2


def test_stronger_gust_pushes_further(msl):
    d = [np.linalg.norm(wind.simulate(msl, [_gust(msl, "parachute", s)])[1]["drift_m"][-1, :2])
         for s in ("light", "strong", "severe")]
    assert d[0] < d[1] < d[2]


def test_powered_descent_barely_moves(msl):
    """Heavy descent stage, thin air, engines correcting: a gust moves it by metres, not hundreds."""
    out, info = wind.simulate(msl, [_gust(msl, "powered_descent", "severe")])
    assert np.linalg.norm(info["drift_m"][:, :2], axis=1).max() < 10
