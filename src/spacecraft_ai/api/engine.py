"""Loads the served models and data once, and answers the API's questions."""
import json
from ast import literal_eval
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..anomaly import model as amodel
from ..faults import inject
from ..faults.experiment import CFG, Scaler, run_detectors, merge_sequences
from ..phases.classifier import EDL_PHASES

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models" / "serving"
DATA = ROOT / "data" / "processed" / "mars_edl"
RESULTS = ROOT / "results"
MISSIONS = {
    "msl": {"name": "Curiosity", "program": "Mars Science Laboratory", "landing_date": "2012-08-06",
            "site": "Gale crater"},
    "m2020": {"name": "Perseverance", "program": "Mars 2020", "landing_date": "2021-02-18",
              "site": "Jezero crater"},
}
SERVED = ["persistence", "lstm_phase_aware"]          # combined = persistence OR phase-aware (best in module 3)
DT = 0.1


def _f(x, nd=4):
    """JSON-safe rounded float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _arr(a, nd=4):
    return [_f(v, nd) for v in a]


class Engine:
    def __init__(self):
        meta = json.loads((MODELS / "meta.json").read_text())
        self.meta = meta
        self.trained_on = meta["trained_on"]
        self.phase_model = joblib.load(MODELS / "phase_classifier.joblib")
        self.scaler = Scaler.__new__(Scaler)
        self.scaler.lo = pd.Series(meta["scaler"]["lo"])
        self.scaler.hi = pd.Series(meta["scaler"]["hi"])
        self.models = {}
        for name, n_feat in meta["n_features"].items():
            variant, ch = name[:-3].split("__")
            net = amodel.LSTMForecaster(n_feat, CFG)
            net.load_state_dict(torch.load(MODELS / name, map_location="cpu"))
            net.eval()
            self.models[(ch, variant)] = net
        missing = [m for m in MISSIONS if not (DATA / f"{m}_edl.csv").exists()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} EDL CSVs in {DATA}. Run scripts/build_mars_dataset.py first.")
        self.edl = {m: pd.read_csv(DATA / f"{m}_edl.csv") for m in MISSIONS}
        self.events = pd.read_csv(DATA / "events.csv")

    # ------------------------------------------------------------------ missions
    def missions(self) -> list[dict]:
        out = []
        for m, info in MISSIONS.items():
            df = self.edl[m]
            td = float(self.events.query("mission == @m and event == 'touchdown'").et_tdb_s.iloc[0])
            ev = [{"event": r.event, "utc": r.utc, "t_s": _f(r.et_tdb_s - td, 2), "altitude_km": _f(r.altitude_km, 3),
                   "speed_m_s": _f(r.speed_rel_mars_m_s, 1)} for r in self.events.query("mission == @m").itertuples()]
            out.append({"id": m, **info, "samples": len(df), "duration_s": _f(len(df) * DT, 1),
                        "entry_speed_m_s": ev[0]["speed_m_s"], "events": ev,
                        "models_trained_on_this_landing": m == self.trained_on})
        return out

    def trajectory(self, mission: str, step: int) -> dict:
        df = self.edl[mission].iloc[::step]
        return {"mission": mission, "step_s": step * DT,
                "t_s": _arr(df.t_minus_touchdown_s, 2),
                "altitude_km": _arr(df.altitude_km, 4),
                "speed_m_s": _arr(df.speed_rel_mars_m_s, 2),
                "vertical_speed_m_s": _arr(df.vertical_speed_m_s, 2),
                "sensed_accel_m_s2": _arr(df.sensed_accel_m_s2, 3),
                "latitude_deg": _arr(df.latitude_deg, 5), "longitude_deg": _arr(df.longitude_deg, 5),
                "phase": df.phase.tolist()}

    # ------------------------------------------------------------------ replay
    def replay(self, mission: str, channel: str, faults: list[dict], random_faults: int, seed: int,
               step: int = 5) -> dict:
        rng = np.random.default_rng(seed)
        df = inject.add_sensor_noise(self.edl[mission], rng)
        scale = self.scaler.range(channel)
        injected: list[inject.Fault] = []
        values = df[channel].to_numpy()
        for f in faults:
            fl = inject.random_fault(df, channel, rng, scale, kind=f["kind"], phase=f["phase"],
                                     min_start=CFG.l_s * 3)
            if f.get("magnitude_frac") is not None:
                fl.magnitude = float(f["magnitude_frac"]) * scale
            injected.append(fl)
        if random_faults:
            _, extra = inject.scenario(df, channel, rng, scale, random_faults, min_start=CFG.l_s * 3)
            injected += extra
        for fl in sorted(injected, key=lambda f: f.start):
            values = inject.apply(values, fl, rng)
        df[channel] = values

        r = run_detectors(df, channel, self.phase_model, self.scaler, self.models, detectors=SERVED)
        combined = merge_sequences(r["persistence"]["sequences"] + r["lstm_phase_aware"]["sequences"])
        t = df.t_minus_touchdown_s.to_numpy()
        pred = r["_phases"]
        true = df.phase.map(EDL_PHASES.index).to_numpy()

        alarms = []
        for s, e in combined:
            e = min(e, len(t) - 1)
            src = [d for d in SERVED if any(a <= e and s <= b for a, b in r[d]["sequences"])]
            hit = [i for i, f in enumerate(injected) if f.start <= e and s <= f.end]
            alarms.append({"start_t": _f(t[s], 2), "end_t": _f(t[e], 2), "phase": EDL_PHASES[pred[s]],
                           "detectors": src, "matches_fault": hit[0] if hit else None,
                           "peak_value": _f(np.max(np.abs(values[s:e + 1])), 3)})
        fault_out = [{"kind": f.kind, "phase": f.phase, "start_t": _f(t[f.start], 2), "end_t": _f(t[f.end], 2),
                      "magnitude": _f(f.magnitude, 3),
                      "detected": any(s <= f.end and f.start <= e for s, e in combined)} for f in injected]

        # anomaly score per row: LSTM smoothed error (first l_s rows have no prediction yet)
        score = np.full(len(df), np.nan)
        e_s = r["lstm_phase_aware"]["e_s"]
        score[CFG.l_s:CFG.l_s + len(e_s)] = e_s
        sl = slice(None, None, step)
        n_false = sum(a["matches_fault"] is None for a in alarms)
        return {
            "mission": mission, "channel": channel, "seed": seed, "step_s": step * DT,
            "trained_on": self.trained_on, "cross_mission": mission != self.trained_on,
            "t_s": _arr(t[sl], 2),
            "altitude_km": _arr(df.altitude_km.to_numpy()[sl], 4),
            "speed_m_s": _arr(df.speed_rel_mars_m_s.to_numpy()[sl], 2),
            "value": _arr(values[sl], 3),
            "anomaly_score": _arr(score[sl], 5),
            "phase_true": [EDL_PHASES[i] for i in true[sl]],
            "phase_pred": [EDL_PHASES[i] for i in pred[sl]],
            "phase_accuracy": _f(np.mean(pred == true), 4),
            "alarms": alarms,
            "faults": fault_out,
            "summary": {"faults_injected": len(injected), "faults_detected": sum(f["detected"] for f in fault_out),
                        "alarms": len(alarms), "false_alarms": n_false},
        }

    # ------------------------------------------------------------------ MSL telemetry (module 1)
    @lru_cache(maxsize=1)
    def _channel_table(self) -> pd.DataFrame:
        df = pd.read_csv(RESULTS / "msl" / "channel_results.csv")
        for c in ("true_sequences", "pred_sequences"):
            df[c] = df[c].apply(literal_eval)
        return df

    def telemetry_channels(self) -> list[dict]:
        return [{"chan_id": r.chan_id, "true_positives": int(r.true_positives), "false_positives": int(r.false_positives),
                 "false_negatives": int(r.false_negatives), "num_true_anoms": int(r.num_true_anoms)}
                for r in self._channel_table().itertuples()]

    def telemetry_channel(self, chan_id: str, max_points: int = 1500) -> dict:
        tab = self._channel_table().set_index("chan_id")
        if chan_id not in tab.index:
            raise KeyError(chan_id)
        path = RESULTS / "msl" / "predictions" / f"{chan_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found. Run scripts/run_anomaly_engine.py --spacecraft MSL.")
        p = np.load(path)
        l_s = 250                                    # module-1 window: prediction i is for test step i + 250
        step = max(1, int(np.ceil(len(p["y"]) / max_points)))
        idx = np.arange(len(p["y"]))[::step] + l_s
        row = tab.loc[chan_id]
        return {"chan_id": chan_id, "step": step, "index": idx.tolist(),
                "value": _arr(p["y"][::step]), "prediction": _arr(p["y_hat"][::step]),
                "error": _arr(p["e_s"][::step], 5),
                "labeled_anomalies": [list(map(int, s)) for s in row.true_sequences],
                "detected_anomalies": [list(map(int, s)) for s in row.pred_sequences],
                "true_positives": int(row.true_positives), "false_positives": int(row.false_positives),
                "false_negatives": int(row.false_negatives)}

    # ------------------------------------------------------------------ headline results
    def summary(self) -> dict:
        def load(p):
            f = RESULTS / p
            return json.loads(f.read_text()) if f.exists() else None
        msl, ph, fa = load("msl/summary.json"), load("phases/summary.json"), load("faults/summary.json")
        out = {}
        if msl:
            out["anomaly_engine"] = {k: _f(msl[k], 3) for k in ("tp", "fp", "fn", "precision", "recall", "f1")}
        if ph:
            out["phase_classifier"] = {k: {"accuracy": _f(v["sequence"]["accuracy"], 3),
                                           "macro_f1": _f(v["sequence"]["macro_f1"], 3)} for k, v in ph.items()}
        if fa:
            out["fault_detection"] = {d: {k: _f(v[k], 3) for k in ("tp", "fp", "fn", "precision", "recall", "f1")}
                                      | {"false_alarms_on_clean": v["false_alarms_on_clean_landings"]}
                                      for d, v in fa["detectors"].items()}
        return out
