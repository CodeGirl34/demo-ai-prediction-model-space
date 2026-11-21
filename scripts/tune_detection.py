"""Choose the anomaly-detection settings on one spacecraft, then score the other one once.

The LSTM predictions stay fixed (they are saved by run_anomaly_engine.py); only the settings
of the dynamic threshold change, so each try takes seconds instead of hours of training.

Why two spacecraft: settings chosen on the same labels they are scored on look better than they
are. Here they are chosen on SMAP (55 channels, 69 anomalies) and tested once on MSL (27 channels,
36 anomalies), which the search never sees.

    # 1. tuning: SMAP only. Writes results/tuning/smap_grid.csv and results/tuning/chosen.json
    python scripts/tune_detection.py tune --preds results/smap_ensemble5_kerasinit --spacecraft SMAP

    # 2. final test, run once: applies chosen.json to MSL. Writes results/tuning/msl_test.json
    python scripts/tune_detection.py test --preds results/msl_ensemble5_kerasinit --spacecraft MSL

Rule fixed before any result was seen: pick the highest F1 (recall is our weak point, and F1
keeps precision in the balance). Ties go to the higher F0.5, then to fewer changes from the
original Telemanom settings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from spacecraft_ai.anomaly import data, metrics, threshold
from spacecraft_ai.anomaly.config import Config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tuning"

# Detection settings of Hundman et al. (2018); the original value is in each list.
GRID = {
    "p": [0.05, 0.08, 0.10, 0.13, 0.16, 0.20],          # pruning: min. relative drop between ranked anomalies
    "error_buffer": [50, 100, 150, 200],                # points added around each flagged error
    "smoothing_perc": [0.02, 0.05, 0.10],               # EWMA span as a share of the threshold window
    "window_size": [20, 30, 50],                        # past batches of errors used for each threshold
}
DEFAULT = {k: getattr(Config(), k) for k in GRID}

_CHANNELS: list = []  # (chan_id, y, y_hat, true_seqs), filled per worker


def _load(preds: Path, spacecraft: str, only: list | None = None) -> list:
    labels = data.load_labels(data.DEFAULT_ROOT, spacecraft)
    if only:
        labels = labels[labels.chan_id.isin(only)]
    out = []
    for row in labels.itertuples():
        f = preds / "predictions" / f"{row.chan_id}.npz"
        if not f.exists():
            raise SystemExit(f"missing predictions for {row.chan_id}: {f}")
        z = np.load(f)
        out.append((row.chan_id, z["y"], z["y_hat"], row.anomaly_sequences))
    return out


def _init(preds: Path, spacecraft: str, only: list | None):
    global _CHANNELS
    _CHANNELS = _load(preds, spacecraft, only)


def score(settings: dict, channels=None, per_channel=False):
    cfg = replace(Config(), **settings)
    tp = fp = fn = 0
    rows = []
    for chan, y, y_hat, true in channels if channels is not None else _CHANNELS:
        det = threshold.detect(y, y_hat, cfg)
        r = metrics.evaluate(det.sequences, true)
        tp, fp, fn = tp + r["true_positives"], fp + r["false_positives"], fn + r["false_negatives"]
        if per_channel:
            rows.append({"chan_id": chan, **r, "true_sequences": true, "pred_sequences": det.sequences})
    s = metrics.summarize(tp, fp, fn)
    return (s, rows) if per_channel else {**settings, **s}


def _changes(row) -> int:
    return sum(row[k] != DEFAULT[k] for k in GRID)


def tune(preds: Path, spacecraft: str, workers: int, only: list | None = None):
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"tuning on {spacecraft}: {len(grid)} settings x {len(_load(preds, spacecraft, only))} channels")
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(preds, spacecraft, only)) as ex:
        res = pd.DataFrame(list(ex.map(score, grid, chunksize=4)))
    res["changes"] = res.apply(_changes, axis=1)
    res = res.sort_values(["f1", "f0.5", "changes"], ascending=[False, False, True]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"{spacecraft.lower()}_grid.csv", index=False)

    best = res.iloc[0]
    default = res[(res[list(GRID)] == pd.Series(DEFAULT)).all(axis=1)].iloc[0]
    chosen = {
        "tuned_on": spacecraft,
        "predictions": _rel(preds),
        "rule": "max F1, then max F0.5, then fewest changes from the original settings",
        "settings": {k: (float(best[k]) if isinstance(DEFAULT[k], float) else int(best[k])) for k in GRID},
        "original_settings": DEFAULT,
        f"{spacecraft.lower()}_tuned": _metrics(best),
        f"{spacecraft.lower()}_original": _metrics(default),
    }
    (OUT / "chosen.json").write_text(json.dumps(chosen, indent=2))
    cols = list(GRID) + ["tp", "fp", "fn", "precision", "recall", "f1", "f0.5"]
    print("\ntop 10 on", spacecraft)
    print(res[cols].head(10).round(3).to_string(index=False))
    print("\noriginal settings on", spacecraft)
    print(default[cols].round(3).to_string())
    print("\nchosen:", chosen["settings"])


def test(preds: Path, spacecraft: str, force: bool):
    chosen_f = OUT / "chosen.json"
    if not chosen_f.exists():
        raise SystemExit("run the tune step first")
    chosen = json.loads(chosen_f.read_text())
    if chosen["tuned_on"] == spacecraft:
        raise SystemExit(f"settings were tuned on {spacecraft}: test on the other spacecraft")
    out_f = OUT / f"{spacecraft.lower()}_test.json"
    if out_f.exists() and not force:
        raise SystemExit(f"{out_f.name} exists: the test set is scored once (use --force to overwrite)")
    channels = _load(preds, spacecraft)
    tuned, rows = score(chosen["settings"], channels, per_channel=True)
    original, _ = score(DEFAULT, channels, per_channel=True)
    result = {"tested_on": spacecraft, "predictions": _rel(preds), "settings": chosen["settings"],
              "tuned": tuned, "original": original}
    out_f.write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(OUT / f"{spacecraft.lower()}_test_channels.csv", index=False)
    print(json.dumps(result, indent=2))


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _metrics(row) -> dict:
    return {k: (int(row[k]) if k in ("tp", "fp", "fn") else round(float(row[k]), 4))
            for k in ["tp", "fp", "fn", "precision", "recall", "f1", "f0.5"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tune", "test"])
    ap.add_argument("--preds", type=Path, required=True, help="run folder with predictions/*.npz")
    ap.add_argument("--spacecraft", choices=["SMAP", "MSL"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--channels", nargs="*", help="quick test on a few channels only")
    ap.add_argument("--out", type=Path, help=f"output folder (default {OUT.relative_to(ROOT)})")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    preds = a.preds if a.preds.is_absolute() else ROOT / a.preds
    tune(preds, a.spacecraft, a.workers, a.channels) if a.step == "tune" else test(preds, a.spacecraft, a.force)
