"""Documented synthetic sensor faults.

The NASA EDL reconstructions are clean, so to *evaluate* an anomaly detector we corrupt a copy of
the data with faults whose type, location and size we record — those records are the labels.

Fault types (common failure modes of inertial / navigation sensors):
    spike    a few samples of a large outlier (bit flip, transient glitch)
    bias     sudden constant offset that persists (calibration jump)
    drift    offset that grows linearly over time (sensor drift)
    stuck    sensor output frozen at its last value
    noise    burst of much higher measurement noise (vibration, interference)
    dropout  sensor output reads zero (lost signal)

Sizes are fractions of the channel's full range R on the *training* landing (the same convention
as the scaled NASA telemetry in module 1), so faults are comparable across channels. For example
a bias of 3-8 % R is 3-8 m/s^2 on sensed acceleration and 90-240 m/s on vertical speed.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

FAULT_TYPES = ["spike", "bias", "drift", "stuck", "noise", "dropout"]

# (min, max) magnitude as a fraction of channel range, and duration in seconds
SPEC = {
    "spike":   dict(mag=(0.10, 0.30), dur=(0.1, 0.5)),
    "bias":    dict(mag=(0.03, 0.08), dur=(3, 20)),
    "drift":   dict(mag=(0.05, 0.12), dur=(10, 30)),
    "stuck":   dict(mag=(0, 0),       dur=(3, 15)),
    "noise":   dict(mag=(0.02, 0.05), dur=(3, 15)),
    "dropout": dict(mag=(0, 0),       dur=(1, 5)),
}

# realistic baseline measurement noise added to *all* data (train and test), 1-sigma
SENSOR_NOISE = {"sensed_accel_m_s2": 0.05, "vertical_speed_m_s": 0.2}


@dataclass
class Fault:
    channel: str
    kind: str
    phase: str
    start: int          # row index
    end: int            # row index, inclusive
    magnitude: float


def add_sensor_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for col, sigma in SENSOR_NOISE.items():
        df[col] = df[col] + rng.normal(0, sigma, len(df))
    return df


def apply(values: np.ndarray, f: Fault, rng: np.random.Generator) -> np.ndarray:
    v = values.copy()
    s, e = f.start, f.end + 1
    n = e - s
    sign = rng.choice([-1, 1])
    if f.kind == "spike":
        v[s:e] += sign * f.magnitude
    elif f.kind == "bias":
        v[s:e] += sign * f.magnitude
    elif f.kind == "drift":
        v[s:e] += sign * f.magnitude * np.linspace(0, 1, n)
    elif f.kind == "stuck":
        v[s:e] = v[s - 1]
    elif f.kind == "noise":
        v[s:e] += rng.normal(0, f.magnitude, n)
    elif f.kind == "dropout":
        v[s:e] = 0.0
    return v


def random_fault(df: pd.DataFrame, channel: str, rng: np.random.Generator, scale: float,
                 kind: str | None = None, phase: str | None = None, dt: float = 0.1,
                 min_start: int = 0) -> Fault:
    """Pick a fault type and phase uniformly, then a random position inside that phase."""
    kind = kind or rng.choice(FAULT_TYPES)
    phase = phase or rng.choice(df["phase"].unique())
    spec = SPEC[kind]
    idx = np.flatnonzero(df["phase"].to_numpy() == phase)
    idx = idx[idx >= min_start]
    n = max(1, int(round(rng.uniform(*spec["dur"]) / dt)))
    n = min(n, max(1, len(idx) - 1))
    start = int(rng.integers(idx[0], idx[-1] - n + 2))
    return Fault(channel, str(kind), str(phase), start, start + n - 1, float(rng.uniform(*spec["mag"]) * scale))


def scenario(df: pd.DataFrame, channel: str, rng: np.random.Generator, scale: float, n_faults: int = 2,
             min_start: int = 0, min_gap: int = 600) -> tuple[pd.DataFrame, list[Fault]]:
    """Copy of `df` with `n_faults` non-overlapping faults on one channel (>= min_gap rows apart)."""
    faults: list[Fault] = []
    while len(faults) < n_faults:
        f = random_fault(df, channel, rng, scale, min_start=min_start)
        if all(f.start > g.end + min_gap or f.end < g.start - min_gap for g in faults):
            faults.append(f)
    out = df.copy()
    v = out[channel].to_numpy()
    for f in sorted(faults, key=lambda f: f.start):
        v = apply(v, f, rng)
    out[channel] = v
    return out, faults


def to_records(faults: list[Fault]) -> list[dict]:
    return [asdict(f) for f in faults]
