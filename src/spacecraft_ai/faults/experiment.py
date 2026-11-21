"""Phase-aware vs phase-blind anomaly detection on Mars landings with injected sensor faults.

Protocol (same spirit as module 2: never test on the landing you trained on)
  1. Add realistic sensor noise to both landings.
  2. Train on the CLEAN training landing:
       - the module-2 phase classifier
       - per monitored channel, an LSTM forecaster (module-1 engine) in two variants:
           phase-aware  inputs = channels + one-hot mission phase   (like command flags in MSL telemetry)
           phase-blind  inputs = channels only
  3. On the OTHER landing, generate many scenarios with injected faults. At test time the phase is
     never taken from the labels: it comes from the phase classifier running on the (faulty) data.
  4. Detect with the module-1 dynamic threshold and score each injected fault as caught / missed;
     also run every detector on the clean test landing to count false alarms.

Detectors: persistence baseline (predict "same as last sample"), phase-blind LSTM, phase-aware LSTM,
and OR-combinations of the baseline with each LSTM (the baseline is strong on sudden faults, the
LSTMs on slow ones).
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from ..anomaly import data as adata, model as amodel, threshold, metrics
from ..anomaly.config import Config
from ..phases import classifier as pclf
from . import inject

CHANNELS = ["sensed_accel_m_s2", "vertical_speed_m_s"]
CONTEXT = ["log_altitude", "speed_rel_mars_m_s"]
DETECTORS = ["persistence", "lstm_phase_blind", "lstm_phase_aware"]
# combined detectors: alarm when EITHER member fires; overlapping alarms are merged into one
COMBINED = {
    "or_persistence_blind": ("persistence", "lstm_phase_blind"),
    "or_persistence_aware": ("persistence", "lstm_phase_aware"),
}
ALL_DETECTORS = DETECTORS + list(COMBINED)

# Pin the original PyTorch-default initialisation if the engine offers a Keras-style option,
# so module-3 results stay reproducible regardless of engine defaults.
_PIN = {"keras_compat": False} if "keras_compat" in Config.__dataclass_fields__ else {}
CFG = Config(
    l_s=60,                       # 6 s of history at 10 Hz
    n_predictions=5,
    epochs=25, patience=6,
    error_buffer=30,              # 3 s padding around detections
    smoothing_perc=10 / (70 * 30),  # EWMA span ~10 samples = 1 s (original 105 assumed slow telemetry)
    **_PIN,
)


class Scaler:
    """Min-max to [-1, 1], fitted on the training landing (module-1 telemetry convention)."""

    def __init__(self, df: pd.DataFrame, cols: list[str]):
        self.lo = df[cols].min()
        self.hi = df[cols].max()

    def __call__(self, df: pd.DataFrame, col: str) -> np.ndarray:
        return (2 * (df[col] - self.lo[col]) / (self.hi[col] - self.lo[col]) - 1).to_numpy(np.float32)

    def range(self, col: str) -> float:
        return float(self.hi[col] - self.lo[col])


def matrix(df: pd.DataFrame, target: str, scaler: Scaler, phases: np.ndarray | None) -> np.ndarray:
    cols = [target] + [c for c in CHANNELS if c != target] + CONTEXT
    arr = np.stack([scaler(df, c) for c in cols], axis=1)
    if phases is not None:
        arr = np.concatenate([arr, np.eye(len(pclf.EDL_PHASES), dtype=np.float32)[phases]], axis=1)
    return arr


def merge_sequences(seqs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of alarm intervals; overlapping or touching intervals become one alarm."""
    out: list[list[int]] = []
    for s, e in sorted(seqs):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def predicted_phases(model, df: pd.DataFrame) -> np.ndarray:
    feats = pclf.add_features(df)
    return pclf.monotonic_decode(model.predict_proba(feats[pclf.FEATURES]))


def run_detectors(df: pd.DataFrame, ch: str, phase_model, scaler: Scaler, models: dict,
                  detectors: list[str] | None = None) -> dict:
    """Phase classification + every detector on one landing (shared by the experiment and the API)."""
    detectors = detectors or DETECTORS
    df = pclf.add_features(df)
    phases = predicted_phases(phase_model, df)
    out = {"_phases": phases}
    for det in detectors:
        arr = matrix(df, ch, scaler, phases if det == "lstm_phase_aware" else None)
        X, y = adata.make_windows(arr, CFG.l_s, CFG.n_predictions)
        if det == "persistence":
            y_hat = arr[CFG.l_s - 1: CFG.l_s - 1 + len(y), 0]
        else:
            y_hat = amodel.predict(models[(ch, det)], X)
        res = threshold.detect(y[:, 0], y_hat, CFG)
        out[det] = {"sequences": res.sequences, "e_s": res.e_s, "y": y[:, 0], "y_hat": y_hat}
    for name, (a, b) in COMBINED.items():
        if a in out and b in out:
            out[name] = {"sequences": merge_sequences(out[a]["sequences"] + out[b]["sequences"])}
    return out


class Experiment:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0, threads: int = 2):
        torch.set_num_threads(threads)
        self.rng = np.random.default_rng(seed)
        self.train = inject.add_sensor_noise(train_df, self.rng).pipe(pclf.add_features)
        self.test = inject.add_sensor_noise(test_df, self.rng)
        self.scaler = Scaler(self.train, CHANNELS + CONTEXT)
        self.phase_model = pclf.train(self.train)
        self.models = {}
        train_phases = self.train["phase"].map(pclf.EDL_PHASES.index).to_numpy()
        for ch in CHANNELS:
            for variant, ph in (("lstm_phase_blind", None), ("lstm_phase_aware", train_phases)):
                X, y = adata.make_windows(matrix(self.train, ch, self.scaler, ph), CFG.l_s, CFG.n_predictions)
                net, info = amodel.train(X, y, replace(CFG, seed=seed))
                self.models[(ch, variant)] = net
                print(f"    trained {variant:17s} {ch:20s} epochs={info['epochs_run']} val_mse={info['best_val_mse']:.2e}",
                      flush=True)

    def detect(self, df: pd.DataFrame, ch: str) -> dict:
        """Run all detectors on one (possibly faulty) landing. Returns detector -> result."""
        return run_detectors(df, ch, self.phase_model, self.scaler, self.models)

    def false_alarms_on_clean(self) -> dict:
        rows = []
        for ch in CHANNELS:
            r = self.detect(self.test, ch)
            ph = r["_phases"]
            for det in ALL_DETECTORS:
                for s, e in r[det]["sequences"]:
                    rows.append({"channel": ch, "detector": det, "start": s, "end": e,
                                 "phase": pclf.EDL_PHASES[ph[min(s, len(ph) - 1)]]})
        return rows

    def run_scenarios(self, n_per_channel: int, faults_per_scenario: int = 2):
        fault_rows, seq_rows, examples = [], [], {}
        for ch in CHANNELS:
            scale = self.scaler.range(ch)
            for k in range(n_per_channel):
                df, faults = inject.scenario(self.test, ch, self.rng, scale, faults_per_scenario,
                                             min_start=CFG.l_s * 3)
                r = self.detect(df, ch)
                for det in ALL_DETECTORS:
                    seqs = r[det]["sequences"]
                    for f in faults:
                        caught = any(s <= f.end and f.start <= e for s, e in seqs)
                        fault_rows.append({**inject.to_records([f])[0], "scenario": k, "detector": det,
                                           "detected": caught})
                    m = metrics.evaluate(seqs, [[f.start, f.end] for f in faults])
                    seq_rows.append({"channel": ch, "scenario": k, "detector": det, **m})
                if k == 0:
                    examples[ch] = (df, faults, r)
                if (k + 1) % 10 == 0:
                    print(f"    {ch}: {k + 1}/{n_per_channel} scenarios", flush=True)
        return pd.DataFrame(fault_rows), pd.DataFrame(seq_rows), examples
