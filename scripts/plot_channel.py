"""Plot telemetry, LSTM prediction, smoothed error and anomalies for one channel.

    python scripts/plot_channel.py M-6            # uses results/msl
    python scripts/plot_channel.py P-1 --spacecraft SMAP
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from ast import literal_eval

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spacecraft_ai.anomaly.config import Config

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chan_id")
    ap.add_argument("--spacecraft", default="MSL")
    a = ap.parse_args()

    res_dir = ROOT / "results" / a.spacecraft.lower()
    p = np.load(res_dir / "predictions" / f"{a.chan_id}.npz")
    row = pd.read_csv(res_dir / "channel_results.csv").set_index("chan_id").loc[a.chan_id]
    true_seqs, pred_seqs = literal_eval(row.true_sequences), literal_eval(row.pred_sequences)
    t = np.arange(len(p["y"])) + Config().l_s     # index in the original test array

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(t, p["y"], lw=0.8, label="telemetry")
    ax1.plot(t, p["y_hat"], lw=0.8, alpha=0.8, label="LSTM prediction")
    ax2.plot(t, p["e_s"], lw=0.8, color="tab:purple", label="smoothed error")
    for ax in (ax1, ax2):
        for s, e in true_seqs:
            ax.axvspan(s, e, color="tab:red", alpha=0.15, label="labeled anomaly" if ax is ax1 and (s, e) == tuple(true_seqs[0]) else None)
        for s, e in pred_seqs:
            ax.axvspan(s, e, ymin=0, ymax=0.06, color="tab:green", alpha=0.9, label="detected" if ax is ax1 and (s, e) == tuple(pred_seqs[0]) else None)
    ax1.set_title(f"{a.spacecraft} channel {a.chan_id}  (TP={row.true_positives} FP={row.false_positives} FN={row.false_negatives})")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("test timestep")
    out = res_dir / "plots"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{a.chan_id}.png", dpi=120)
    print(out / f"{a.chan_id}.png")
