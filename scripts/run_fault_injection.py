"""Module 3: inject sensor faults into Mars landings and compare anomaly detectors.

    python scripts/build_mars_dataset.py     # first
    python scripts/run_fault_injection.py --scenarios 40
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spacecraft_ai.anomaly import metrics
from spacecraft_ai.faults.experiment import Experiment, DETECTORS, CFG
from spacecraft_ai.faults.inject import FAULT_TYPES
from spacecraft_ai.phases.classifier import EDL_PHASES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "mars_edl"
OUT = ROOT / "results" / "faults"
NAMES = {"m2020": "Perseverance", "msl": "Curiosity"}
LABEL = {"persistence": "persistence baseline", "lstm_phase_blind": "LSTM phase-blind",
         "lstm_phase_aware": "LSTM phase-aware"}


def plot_example(df, faults, r, ch, title, path):
    t = np.arange(len(df)) * 0.1
    l_s = CFG.l_s
    fig, axes = plt.subplots(len(DETECTORS) + 1, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(t, df[ch], lw=0.6, color="black")
    axes[0].set_ylabel(ch.replace("_", " "), fontsize=8)
    for f in faults:
        for ax in axes:
            ax.axvspan(f.start * 0.1, f.end * 0.1, color="tab:red", alpha=0.18)
        axes[0].text(f.start * 0.1, axes[0].get_ylim()[1], f" {f.kind} ({f.phase})", fontsize=8,
                     color="tab:red", va="top")
    for ax, det in zip(axes[1:], DETECTORS):
        e = r[det]["e_s"]
        ax.plot(t[l_s:l_s + len(e)], e, lw=0.6, color="tab:purple")
        for s, e2 in r[det]["sequences"]:
            ax.axvspan(s * 0.1, e2 * 0.1, ymin=0, ymax=0.12, color="tab:green")
        ax.set_ylabel(LABEL[det], fontsize=8)
        ax.set_yscale("log")
    axes[-1].set_xlabel("seconds since start of EDL data")
    axes[0].set_title(title + "   (red = injected fault, green = detection)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=40, help="scenarios per channel per direction")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    edl = {m: pd.read_csv(DATA / f"{m}_edl.csv") for m in NAMES}

    all_faults, all_seqs, all_fa = [], [], []
    for train_m, test_m in (("m2020", "msl"), ("msl", "m2020")):
        tag = f"{NAMES[train_m]} -> {NAMES[test_m]}"
        print(f"\n=== train {tag}", flush=True)
        exp = Experiment(edl[train_m], edl[test_m], seed=0 if train_m == "m2020" else 1, threads=a.threads)
        fa = pd.DataFrame(exp.false_alarms_on_clean()).assign(direction=tag)
        faults, seqs, examples = exp.run_scenarios(a.scenarios)
        all_faults.append(faults.assign(direction=tag)); all_seqs.append(seqs.assign(direction=tag)); all_fa.append(fa)
        for ch, (df, fl, r) in examples.items():
            plot_example(df, fl, r, ch, f"Trained on {NAMES[train_m]}, faults injected into {NAMES[test_m]}: {ch}",
                         OUT / f"example_{train_m}_to_{test_m}_{ch}.png")

    faults = pd.concat(all_faults); seqs = pd.concat(all_seqs)
    fa = pd.concat(all_fa) if any(len(x) for x in all_fa) else pd.DataFrame(columns=["detector", "phase", "direction"])
    faults.to_csv(OUT / "injected_faults.csv", index=False)
    fa.to_csv(OUT / "false_alarms_clean.csv", index=False)

    summary = {"config": CFG.to_dict(), "n_faults_per_detector": int(len(faults) // len(DETECTORS)), "detectors": {}}
    for det in DETECTORS:
        s = seqs[seqs.detector == det]
        f = faults[faults.detector == det]
        tot = metrics.summarize(int(s.true_positives.sum()), int(s.false_positives.sum()), int(s.false_negatives.sum()))
        summary["detectors"][det] = {
            **{k: round(float(v), 3) for k, v in tot.items()},
            "false_alarms_on_clean_landings": int((fa.detector == det).sum()) if len(fa) else 0,
            "recall_by_fault_type": f.groupby("kind").detected.mean().reindex(FAULT_TYPES).round(3).to_dict(),
            "recall_by_phase": f.groupby("phase").detected.mean().reindex(EDL_PHASES).round(3).to_dict(),
            "recall_by_channel": f.groupby("channel").detected.mean().round(3).to_dict(),
            "recall_by_direction": f.groupby("direction").detected.mean().round(3).to_dict(),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== RESULTS (both directions, both channels)")
    for det, r in summary["detectors"].items():
        print(f"{LABEL[det]:22s} recall={r['recall']:.3f} precision={r['precision']:.3f} F1={r['f1']:.3f} "
              f"false-alarms-on-clean={r['false_alarms_on_clean_landings']}")
        print(f"{'':22s} by type  {r['recall_by_fault_type']}")
        print(f"{'':22s} by phase {r['recall_by_phase']}")
    print(f"\nsaved {OUT}")
