"""Check our dynamic-threshold code against the original 2018 Telemanom run.

The Kaggle archive ships the original LSTM predictions (y_hat) and run log.
Feeding those predictions into our threshold code should reproduce the
per-channel TP/FP/FN in the original log.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import re
import sys
from pathlib import Path

import numpy as np

from spacecraft_ai.anomaly import data, threshold, metrics
from spacecraft_ai.anomaly.config import Config

RUN = data._data_dir(data.DEFAULT_ROOT) / "2018-05-19_15.00.10"
spacecraft = sys.argv[1] if len(sys.argv) > 1 else "MSL"

log = RUN.joinpath("params.log").read_text()
orig = {m[0]: tuple(map(int, m[1:])) for m in re.findall(r"Stream # \d+: (\S+)\n.*?\n.*?TP: (\d+)  FP: (\d+)  FN: (\d+)", log)}

cfg = Config()
labels = data.load_labels(spacecraft=spacecraft)
match, tot_ours, tot_orig = 0, np.zeros(3, int), np.zeros(3, int)
for row in labels.itertuples():
    _, test = data.load_channel(row.chan_id)
    y_hat = np.load(RUN / "y_hat" / f"{row.chan_id}.npy")
    y = test[cfg.l_s:cfg.l_s + len(y_hat), 0]
    det = threshold.detect(y, y_hat, cfg, fill_initial=row.chan_id != "C-2")  # mirrors reference special case
    r = metrics.evaluate(det.sequences, row.anomaly_sequences)
    ours = (r["true_positives"], r["false_positives"], r["false_negatives"])
    o = orig.get(row.chan_id)
    ok = ours == o
    match += ok
    tot_ours += ours
    tot_orig += o
    print(f"{row.chan_id:6s} ours TP/FP/FN={ours}  original={o}  {'OK' if ok else 'DIFF'}")
print(f"\n{match}/{len(labels)} channels identical")
fmt = lambda d: {k: round(float(v), 3) for k, v in d.items()}
print("ours    :", fmt(metrics.summarize(*map(int, tot_ours))))
print("original:", fmt(metrics.summarize(*map(int, tot_orig))))
