"""Mission-phase classifier for Mars entry, descent and landing.

Inputs are physical state variables that mean the same thing on any Mars lander (no clock time,
no mission id), so a model trained on one landing can be tested on a different spacecraft.
All rolling features look only at the past, so the classifier could run in real time.

Two outputs:
  * per-sample prediction (gradient-boosted trees)
  * sequence-constrained prediction: phases only move forward (approach -> entry -> ...),
    decoded with dynamic programming over the per-sample probabilities.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

EDL_PHASES = ["approach", "entry", "parachute", "powered_descent", "sky_crane"]
BASE_FEATURES = ["altitude_km", "speed_rel_mars_m_s", "vertical_speed_m_s",
                 "flight_path_angle_deg", "sensed_accel_m_s2"]


def add_features(df: pd.DataFrame, dt: float = 0.1) -> pd.DataFrame:
    df = df.copy()
    w = max(1, int(round(2.0 / dt)))                          # 2-second causal window
    acc = df["sensed_accel_m_s2"]
    df["accel_mean_2s"] = acc.rolling(w, min_periods=1).mean()
    df["accel_std_2s"] = acc.rolling(w, min_periods=1).std().fillna(0)
    df["log_altitude"] = np.log10(df["altitude_km"].clip(lower=1e-3))
    df["speed_change_2s"] = df["speed_rel_mars_m_s"].diff(w).fillna(0)
    return df


FEATURES = BASE_FEATURES + ["accel_mean_2s", "accel_std_2s", "log_altitude", "speed_change_2s"]


def train(df: pd.DataFrame, seed: int = 0) -> HistGradientBoostingClassifier:
    y = df["phase"].map(EDL_PHASES.index).to_numpy()
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           class_weight="balanced", random_state=seed)
    model.fit(df[FEATURES], y)
    return model


def monotonic_decode(proba: np.ndarray) -> np.ndarray:
    """Most likely label sequence that never moves backwards through the phases."""
    n, k = proba.shape
    logp = np.log(np.clip(proba, 1e-12, 1))
    score = np.full((n, k), -np.inf)
    back = np.zeros((n, k), dtype=int)
    score[0] = logp[0]
    for t in range(1, n):
        best_prev = np.maximum.accumulate(score[t - 1])          # best over phases <= j
        arg_prev = np.array([np.argmax(score[t - 1][: j + 1]) for j in range(k)])
        score[t] = best_prev + logp[t]
        back[t] = arg_prev
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def transition_times(labels: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Time of the first sample of each phase (after the first)."""
    out = {}
    for j, ph in enumerate(EDL_PHASES[1:], 1):
        idx = np.flatnonzero(labels >= j)
        out[ph] = float(t[idx[0]]) if idx.size else np.nan
    return out


def evaluate(model, df: pd.DataFrame) -> dict:
    y = df["phase"].map(EDL_PHASES.index).to_numpy()
    proba = model.predict_proba(df[FEATURES])
    raw = proba.argmax(1)
    seq = monotonic_decode(proba)
    t = df["t_minus_touchdown_s"].to_numpy()
    true_tr = transition_times(y, t)
    res = {}
    for name, pred in (("per_sample", raw), ("sequence", seq)):
        tr = transition_times(pred, t)
        res[name] = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "per_phase_f1": dict(zip(EDL_PHASES, f1_score(y, pred, average=None, labels=range(5)).round(3))),
            "transition_error_s": {ph: round(tr[ph] - true_tr[ph], 2) for ph in tr},
            "confusion": confusion_matrix(y, pred, labels=range(5)).tolist(),
        }
    res["_pred_seq"] = seq
    res["_pred_raw"] = raw
    return res
