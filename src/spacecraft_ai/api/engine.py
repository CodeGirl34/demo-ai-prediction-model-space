"""Loads the served models and data once, and answers the API's questions."""
import copy
import json
from ast import literal_eval
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..anomaly import model as amodel
from ..faults import inject, wind
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
               step: int = 5, gusts: list[dict] | None = None) -> dict:
        rng = np.random.default_rng(seed)
        base = self.edl[mission]
        # wind gusts move the spacecraft itself (all sensors agree); own random stream, so adding a
        # gust does not change where the sensor faults land
        blown = [wind.random_gust(base, np.random.default_rng([seed, 101, i]), g["phase"], g.get("strength", "strong"))
                 for i, g in enumerate(gusts or [])]
        moved, flight = wind.simulate(base, blown)
        df = inject.add_sensor_noise(base, rng)
        calm = df.copy()                             # the same landing without the wind (see below)
        nasa = df[channel].to_numpy().copy()         # NASA's landing as flown: no wind, no faults
        for col in wind.PERTURBED:
            df[col] = df[col] + (moved[col] - base[col])
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
        healthy = values.copy()                      # what a working sensor would have read
        rng_calm = copy.deepcopy(rng)                # the same fault draws for the no-wind run
        for fl in sorted(injected, key=lambda f: f.start):
            values = inject.apply(values, fl, rng)
        df[channel] = values
        change = np.abs(values - healthy)

        # each gust's window: the gust and up to 30 s after it, while the reading is still more
        # than 3 sigma away from NASA's. Alarms in it are the gust, not false alarms.
        wind_dev = np.abs(healthy - nasa)
        thresh = 3 * inject.SENSOR_NOISE.get(channel, 0.1)
        gust_win = []
        for k, g in enumerate(sorted(blown, key=lambda g: g.start)):
            stop = min([h.start for h in blown if h.start > g.start] + [g.end + int(30 / DT), len(df)])
            above = np.flatnonzero(wind_dev[g.start:stop] > thresh)
            gust_win.append((g, g.start, max(g.end, g.start + int(above[-1]) if len(above) else g.end)))

        r = run_detectors(df, channel, self.phase_model, self.scaler, self.models, detectors=SERVED)
        combined = merge_sequences(r["persistence"]["sequences"] + r["lstm_phase_aware"]["sequences"])
        # The AI noticed a gust only if it raises an alarm that it does not raise on the same landing,
        # with the same sensor noise and faults, without the wind. Otherwise an alarm the AI would
        # have raised anyway would count as noticing the gust.
        calm_alarms = []
        if blown:
            calm_values = calm[channel].to_numpy().copy()
            for fl in sorted(injected, key=lambda f: f.start):
                calm_values = inject.apply(calm_values, fl, rng_calm)
            calm[channel] = calm_values
            rc = run_detectors(calm, channel, self.phase_model, self.scaler, self.models, detectors=SERVED)
            calm_alarms = merge_sequences(rc["persistence"]["sequences"] + rc["lstm_phase_aware"]["sequences"])
        t = df.t_minus_touchdown_s.to_numpy()
        pred = r["_phases"]
        true = df.phase.map(EDL_PHASES.index).to_numpy()

        alarms = []
        for s, e in combined:
            e = min(e, len(t) - 1)
            src = [d for d in SERVED if any(a <= e and s <= b for a, b in r[d]["sequences"])]
            hit = [i for i, f in enumerate(injected) if f.start <= e and s <= f.end]
            gust = [i for i, (_, a, b) in enumerate(gust_win) if a <= e and s <= b]
            anyway = any(a <= e and s <= b for a, b in calm_alarms)   # also there without the wind
            alarms.append({"start_t": _f(t[s], 2), "end_t": _f(t[e], 2), "phase": EDL_PHASES[pred[s]],
                           "detectors": src, "matches_fault": hit[0] if hit else None,
                           "matches_gust": gust[0] if gust and not hit and not anyway else None,
                           "peak_value": _f(np.max(np.abs(values[s:e + 1])), 3)})
        fault_out = [{"kind": f.kind, "phase": f.phase, "start_t": _f(t[f.start], 2), "end_t": _f(t[f.end], 2),
                      "magnitude": _f(f.magnitude, 3),
                      # how much the reading actually moved: a stuck sensor or a dropout can change almost
                      # nothing when the real value is flat or already near zero
                      "max_change": _f(change[f.start:f.end + 1].max(), 3),
                      "max_change_frac": _f(change[f.start:f.end + 1].max() / scale, 5),
                      "detected": any(s <= f.end and f.start <= e for s, e in combined)} for f in injected]

        # anomaly score per row: LSTM smoothed error (first l_s rows have no prediction yet)
        score = np.full(len(df), np.nan)
        e_s = r["lstm_phase_aware"]["e_s"]
        score[CFG.l_s:CFG.l_s + len(e_s)] = e_s
        # thin to one row per `step`, but inside each bucket keep the row where the fault moved the reading
        # most, so a 0.1 s spike is not dropped from the curve (all rows are real samples)
        n = len(df)
        buckets = np.arange(0, n, step)
        sl = np.array([b + int(np.argmax(change[b:b + step])) if change[b:b + step].max() > 0 else b
                       for b in buckets])
        n_false = sum(a["matches_fault"] is None and a["matches_gust"] is None for a in alarms)
        gust_out = []
        for k, (g, a, b) in enumerate(gust_win):
            _, alone = wind.simulate(base, [g])     # this gust's own push, if there are several
            h = np.linalg.norm(alone["drift_m"][:, :2], axis=1)
            gust_out.append({
                "phase": g.phase, "strength": g.strength, "peak_wind_m_s": _f(g.peak_m_s, 1),
                "vertical_m_s": _f(g.vertical_m_s, 1), "direction_deg": _f(g.direction_deg, 0),
                "start_t": _f(t[g.start], 2), "end_t": _f(t[g.end], 2), "effect_end_t": _f(t[min(b, len(t) - 1)], 2),
                "max_drift_m": _f(h.max(), 1), "drift_at_touchdown_m": _f(h[-1], 1),
                "max_change": _f(wind_dev[a:b + 1].max(), 3),
                "noticed": any(al["matches_gust"] == k for al in alarms)})
        wind_out = None
        if blown:
            wf, dr, sw = flight["wind"], flight["drift_m"], flight["swing"]
            wind_out = {"e": _arr(wf[sl, 0], 2), "n": _arr(wf[sl, 1], 2), "u": _arr(wf[sl, 2], 2),
                        "drift_e_m": _arr(dr[sl, 0], 2), "drift_n_m": _arr(dr[sl, 1], 2), "drift_u_m": _arr(dr[sl, 2], 2),
                        "swing_e": _arr(sw[sl, 0], 4), "swing_n": _arr(sw[sl, 1], 4)}
        return {
            "mission": mission, "channel": channel, "seed": seed, "step_s": step * DT,
            "trained_on": self.trained_on, "cross_mission": mission != self.trained_on,
            "t_s": _arr(t[sl], 2),
            "altitude_km": _arr(df.altitude_km.to_numpy()[sl], 4),
            "speed_m_s": _arr(df.speed_rel_mars_m_s.to_numpy()[sl], 2),
            "value": _arr(values[sl], 3),
            "value_healthy": _arr(healthy[sl], 3),
            "value_nasa": _arr(nasa[sl], 3),
            "channel_range": _f(scale, 3),
            "anomaly_score": _arr(score[sl], 5),
            "phase_true": [EDL_PHASES[i] for i in true[sl]],
            "phase_pred": [EDL_PHASES[i] for i in pred[sl]],
            "phase_accuracy": _f(np.mean(pred == true), 4),
            "alarms": alarms,
            "faults": fault_out,
            "gusts": gust_out,
            "wind": wind_out,
            "summary": {"faults_injected": len(injected), "faults_detected": sum(f["detected"] for f in fault_out),
                        "gusts": len(gust_out), "gusts_noticed": sum(g["noticed"] for g in gust_out),
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
