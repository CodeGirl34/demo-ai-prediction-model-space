"""Wind gusts: a real push on the spacecraft, not a broken sensor.

A sensor fault (inject.py) corrupts one reading while the spacecraft flies NASA's path. A gust is
the opposite: every sensor still works, but the spacecraft really moves differently, so altitude,
speed, vertical speed, flight-path angle and sensed acceleration all change together.

How a gust moves the spacecraft
  * Gust shape: the "1 - cosine" discrete gust used in aircraft certification (EASA CS-25.341):
    the wind rises smoothly to its peak and falls back to zero over the gust's duration. It blows
    horizontally in a random direction, with a smaller vertical part (up- or downdraft).
  * Entry and parachute (no engines): the only push is aerodynamic drag, a = -k |u| u, where u is
    the velocity relative to the air. k comes from NASA's own data at each sample,
    k = sensed acceleration / speed^2, so the model uses the drag the real vehicle felt. A gust
    changes u, so it changes the drag, and the vehicle is dragged along with the moving air.
  * No gusts in approach: above ~125 km there is no air to push anything.
  * Powered descent and sky crane: the descent stage with the rover is heavy and Mars's air is
    thin, so drag is tiny (stated assumptions below). Guidance then works against the push:
    - It cancels any sideways speed within a few seconds.
    - It brings the altitude back to the planned profile over the ~40 s of powered descent.
    - It does not fly back to the original touchdown point, so the rover lands where the wind
      left it.
  * The sensors read the new motion correctly. The sensed acceleration also includes the extra
    drag and the guidance thrust.

This is a first-order model for a demo, not a flight simulator. It has no parachute pendulum
dynamics, no lift changes in entry, and no winds already contained in NASA's reconstruction.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

GUST_PHASES = ["entry", "parachute", "powered_descent", "sky_crane"]
GUST_STRENGTHS = {"light": 10.0, "strong": 20.0, "severe": 35.0}    # peak horizontal wind, m/s
DURATION_S = (15.0, 30.0)
VERTICAL_SHARE = 0.3            # peak up/downdraft as a share of the horizontal peak

# Powered descent vehicle (descent stage + rover): rounded public figures, stated as assumptions
PDV_MASS_KG = 1900.0
PDV_DRAG_AREA_M2 = 9.0          # drag coefficient x frontal area (about 1.0 x 3 m x 3 m)
RHO_SITE_KG_M3 = 0.017          # air density near the landing sites (about 1.5 % of Earth's)
SCALE_HEIGHT_M = 11_100.0
GUIDANCE_TAU_S = 4.0            # powered descent cancels sideways speed on this time scale
GUIDANCE_OMEGA = 0.15           # rad/s: critically damped return to the planned altitude, spread
                                # over the ~40 s of powered descent instead of one hard kick
GUIDANCE_RAMP_S = 3.0           # guidance corrections fade in after back shell separation

OMEGA_MARS = 7.088218e-5        # rad/s
GM_MARS_KM3 = 42828.37          # km^3/s^2

# columns a gust changes (everything else, e.g. timestamps and phase labels, stays)
PERTURBED = ["altitude_km", "speed_rel_mars_m_s", "vertical_speed_m_s", "flight_path_angle_deg",
             "sensed_accel_m_s2", "latitude_deg", "longitude_deg"]


@dataclass
class Gust:
    phase: str
    strength: str
    start: int              # row index
    end: int                # row index, inclusive
    peak_m_s: float         # horizontal peak
    direction_deg: float    # where the wind blows towards, clockwise from north
    vertical_m_s: float     # peak vertical part, + = updraft


def random_gust(df: pd.DataFrame, rng: np.random.Generator, phase: str, strength: str = "strong",
                dt: float = 0.1) -> Gust:
    """A gust at a random time inside `phase`; shortened if the phase is shorter than the gust."""
    if phase not in GUST_PHASES:
        raise ValueError(f"no air to blow in phase {phase!r}: gusts need one of {GUST_PHASES}")
    idx = np.flatnonzero(df["phase"].to_numpy() == phase)
    n = int(round(rng.uniform(*DURATION_S) / dt))
    n = min(n, len(idx) - 1)
    start = int(rng.integers(idx[0], idx[-1] - n + 2))
    peak = GUST_STRENGTHS[strength]
    return Gust(phase, strength, start, start + n - 1, peak, float(rng.uniform(0, 360)),
                float(rng.choice([-1, 1]) * VERTICAL_SHARE * peak))


def wind_enu(n_rows: int, gusts: list[Gust]) -> np.ndarray:
    """Wind velocity (east, north, up) in m/s at every row: 1 - cosine shape per gust."""
    w = np.zeros((n_rows, 3))
    for g in gusts:
        k = np.arange(g.end - g.start + 1)
        shape = 0.5 * (1 - np.cos(2 * np.pi * k / max(1, len(k) - 1)))
        az = np.radians(g.direction_deg)
        w[g.start:g.end + 1] += np.outer(shape, [g.peak_m_s * np.sin(az), g.peak_m_s * np.cos(az), g.vertical_m_s])
    return w


def _local_frame(df: pd.DataFrame):
    """Mars-fixed position (km), and the east/north/up unit vectors at every row."""
    lat, lon = np.radians(df["latitude_deg"].to_numpy()), np.radians(df["longitude_deg"].to_numpy())
    rad = df["radius_km"].to_numpy()
    cl, sl, co, so = np.cos(lat), np.sin(lat), np.cos(lon), np.sin(lon)
    r = np.stack([rad * cl * co, rad * cl * so, rad * sl], 1)
    e = np.stack([-so, co, np.zeros_like(co)], 1)
    nn = np.stack([-sl * co, -sl * so, cl], 1)
    u = np.stack([cl * co, cl * so, sl], 1)
    return r, e, nn, u


def nominal_motion(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """NASA's velocity relative to the ground and its sensed acceleration, both (east, north, up), SI units.

    Velocity is the time derivative of the Mars-fixed position. Sensed acceleration is the
    rotating-frame acceleration minus gravity. Its size matches the `sensed_accel_m_s2` column
    within ~0.04 m/s^2 from parachute to touchdown (tests/test_wind.py).
    """
    t = df["et_tdb_s"].to_numpy()
    r, e, nn, u = _local_frame(df)
    v = np.gradient(r, t, axis=0)
    a = np.gradient(v, t, axis=0)
    w = np.array([0.0, 0.0, OMEGA_MARS])
    g = -GM_MARS_KM3 * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    sensed = a + 2 * np.cross(w, v) + np.cross(w, np.cross(w, r)) - g
    to_enu = lambda x: np.stack([np.einsum("ij,ij->i", x, b) for b in (e, nn, u)], 1) * 1000
    return to_enu(v), to_enu(sensed)


def _drag_k(phase: str, sensed: float, v_nom: np.ndarray, alt_m: float) -> float:
    if phase in ("entry", "parachute"):
        return sensed / max(float(v_nom @ v_nom), 1.0)         # the drag NASA's vehicle really felt
    if phase in ("powered_descent", "sky_crane"):
        rho = RHO_SITE_KG_M3 * np.exp(-max(alt_m, 0.0) / SCALE_HEIGHT_M)
        return rho * PDV_DRAG_AREA_M2 / (2 * PDV_MASS_KG)
    return 0.0                                                  # approach: no air


def simulate(df: pd.DataFrame, gusts: list[Gust], dt: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """Fly NASA's landing again with the gusts, return the moved trajectory and what happened.

    The second value holds per-row arrays (east, north, up):
    * `wind`: the wind
    * `drift_m`: how far the spacecraft is from NASA's path
    * `dv`: velocity change
    * `da`: change in sensed acceleration
    * `swing`: how far the sensed force tilts away from NASA's direction, in radians. That is the
      parachute's line under drag, or the descent stage's thrust axis.
    """
    n = len(df)
    wind = wind_enu(n, gusts)
    dr, dv, da = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    out = df.copy()
    if not gusts:
        return out, {"wind": wind, "drift_m": dr, "dv": dv, "da": da, "swing": np.zeros((n, 2))}

    v_nom, a_nom = nominal_motion(df)
    phase = df["phase"].to_numpy()
    sensed = df["sensed_accel_m_s2"].to_numpy()
    alt_m = df["altitude_km"].to_numpy() * 1000
    r_now, v_now = np.zeros(3), np.zeros(3)
    powered_s = 0.0
    for i in range(min(g.start for g in gusts), n):
        ph = phase[i]
        k = _drag_k(ph, sensed[i], v_nom[i], alt_m[i])
        u0 = v_nom[i]                         # air-relative velocity with no wind
        u1 = v_nom[i] + v_now - wind[i]       # ... with the gust and the response so far
        acc = -k * (np.linalg.norm(u1) * u1 - np.linalg.norm(u0) * u0)
        if ph in ("powered_descent", "sky_crane"):
            gain = min(1.0, powered_s / GUIDANCE_RAMP_S)
            powered_s += dt
            acc[:2] -= gain * v_now[:2] / GUIDANCE_TAU_S                           # cancel sideways speed
            acc[2] -= gain * (2 * GUIDANCE_OMEGA * v_now[2] + GUIDANCE_OMEGA ** 2 * r_now[2])  # back to the profile
        da[i] = acc
        v_now = v_now + acc * dt
        r_now = r_now + v_now * dt
        dv[i], dr[i] = v_now, r_now

    v1 = v_nom + dv
    speed0, speed1 = np.linalg.norm(v_nom, axis=1), np.linalg.norm(v1, axis=1)
    out["speed_rel_mars_m_s"] = df["speed_rel_mars_m_s"] + (speed1 - speed0)
    out["vertical_speed_m_s"] = df["vertical_speed_m_s"] + dv[:, 2]
    out["flight_path_angle_deg"] = np.degrees(np.arcsin(np.clip(
        out["vertical_speed_m_s"] / np.maximum(out["speed_rel_mars_m_s"], 1e-6), -1, 1)))
    out["altitude_km"] = df["altitude_km"] + dr[:, 2] / 1000
    out["sensed_accel_m_s2"] = df["sensed_accel_m_s2"] + (np.linalg.norm(a_nom + da, axis=1)
                                                          - np.linalg.norm(a_nom, axis=1))
    rad_m = df["radius_km"].to_numpy() * 1000
    out["latitude_deg"] = df["latitude_deg"] + np.degrees(dr[:, 1] / rad_m)
    out["longitude_deg"] = df["longitude_deg"] + np.degrees(dr[:, 0] / (rad_m * np.cos(np.radians(df["latitude_deg"]))))

    # tilt of the sensed force (parachute line or thrust axis) away from NASA's direction
    a0 = a_nom / np.maximum(np.linalg.norm(a_nom, axis=1), 1e-6)[:, None]
    a1 = (a_nom + da) / np.maximum(np.linalg.norm(a_nom + da, axis=1), 1e-6)[:, None]
    swing = np.clip((a1 - a0)[:, :2], -0.6, 0.6)
    # with almost no force (approach, the moment after back shell separation) the direction means little
    swing *= np.clip((sensed - 0.5) / 2.5, 0, 1)[:, None]
    return out, {"wind": wind, "drift_m": dr, "dv": dv, "da": da, "swing": swing}


def to_records(gusts: list[Gust]) -> list[dict]:
    return [asdict(g) for g in gusts]
