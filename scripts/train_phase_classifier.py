"""Train the EDL phase classifier on one Mars landing and test it on the other.

    python scripts/build_mars_dataset.py        # first
    python scripts/train_phase_classifier.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spacecraft_ai.phases import classifier as clf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "mars_edl"
OUT = ROOT / "results" / "phases"
NAMES = {"m2020": "Perseverance", "msl": "Curiosity"}
COLORS = dict(zip(clf.EDL_PHASES, ["#9e9e9e", "#e4572e", "#29a3d6", "#f3a712", "#4caf50"]))


def plot(df, res, title, path):
    t = df["t_minus_touchdown_s"].to_numpy()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(t, df["altitude_km"], color="black", lw=1)
    ax.set_yscale("log"); ax.set_ylim(5e-3, 2e3); ax.set_ylabel("altitude above landing site (km)")
    a2 = ax.twinx(); a2.plot(t, df["sensed_accel_m_s2"] / 9.80665, color="tab:purple", lw=0.6, alpha=0.7)
    a2.set_ylabel("sensed deceleration (g)", color="tab:purple")
    y = df["phase"].map(clf.EDL_PHASES.index).to_numpy()
    for row, lab, name in ((1, y, "actual (NASA timeline)"), (0, res["_pred_seq"], "predicted")):
        for j, ph in enumerate(clf.EDL_PHASES):
            m = lab == j
            if m.any():
                bx.fill_between(t, row, row + 0.8, where=m, color=COLORS[ph], step="mid",
                                label=ph if row == 1 else None)
        bx.text(t[0] - 5, row + 0.4, name, ha="right", va="center", fontsize=9)
    bx.set_yticks([]); bx.set_ylim(-0.1, 1.9); bx.set_xlabel("seconds before touchdown")
    bx.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    s = res["sequence"]
    ax.set_title(f"{title}   accuracy {s['accuracy']:.3f}, macro-F1 {s['macro_f1']:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = {m: clf.add_features(pd.read_csv(DATA / f"{m}_edl.csv")) for m in NAMES}
    summary = {}
    for train_m, test_m in (("m2020", "msl"), ("msl", "m2020")):
        model = clf.train(data[train_m])
        res = clf.evaluate(model, data[test_m])
        key = f"train_{train_m}_test_{test_m}"
        summary[key] = {k: v for k, v in res.items() if not k.startswith("_")}
        plot(data[test_m], res, f"Trained on {NAMES[train_m]}, tested on {NAMES[test_m]}", OUT / f"{key}.png")
        print(f"\n=== train {NAMES[train_m]} -> test {NAMES[test_m]}")
        for kind in ("per_sample", "sequence"):
            r = res[kind]
            print(f"  {kind:10s} acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f}  F1={r['per_phase_f1']}")
            print(f"  {'':10s} phase-start timing error (s): {r['transition_error_s']}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nsaved {OUT}")
