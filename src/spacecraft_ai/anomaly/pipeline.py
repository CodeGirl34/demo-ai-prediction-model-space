"""End-to-end: train an LSTM per channel, predict the test set, detect and score anomalies."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, model, threshold, metrics
from .config import Config


def run_channel(chan_id: str, true_seqs: list, cfg: Config, root: Path, model_dir: Path | None, verbose=False) -> dict:
    t0 = time.time()
    train_arr, test_arr = data.load_channel(chan_id, root)
    X_tr, y_tr = data.make_windows(train_arr, cfg.l_s, cfg.n_predictions)
    X_te, y_te = data.make_windows(test_arr, cfg.l_s, cfg.n_predictions)

    net, info = model.train(X_tr, y_tr, cfg, verbose=verbose)
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), model_dir / f"{chan_id}.pt")

    y_hat = model.predict(net, X_te)
    y = y_te[:, 0]
    det = threshold.detect(y, y_hat, cfg)
    res = metrics.evaluate(det.sequences, true_seqs)
    return {
        "chan_id": chan_id,
        **res,
        "num_true_anoms": len(true_seqs),
        "true_sequences": true_seqs,
        "pred_sequences": det.sequences,
        "scores": det.scores,
        "normalized_error": float(np.mean(np.abs(y - y_hat)) / np.ptp(y)) if np.ptp(y) else 0.0,
        "epochs_run": info["epochs_run"],
        "best_val_mse": info["best_val_mse"],
        "seconds": round(time.time() - t0, 1),
        "_y": y, "_y_hat": y_hat, "_e_s": det.e_s,
    }


def run(spacecraft: str, cfg: Config, root: Path, out_dir: Path, channels: list[str] | None = None, verbose=False):
    labels = data.load_labels(root, spacecraft)
    if channels:
        labels = labels[labels.chan_id.isin(channels)]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    rows = []
    for k, row in enumerate(labels.itertuples(), 1):
        r = run_channel(row.chan_id, row.anomaly_sequences, cfg, root, out_dir / "models", verbose)
        np.savez_compressed(pred_dir / f"{row.chan_id}.npz", y=r.pop("_y"), y_hat=r.pop("_y_hat"), e_s=r.pop("_e_s"))
        rows.append(r)
        print(f"[{k:2d}/{len(labels)}] {row.chan_id:6s} TP={r['true_positives']} FP={r['false_positives']} "
              f"FN={r['false_negatives']}  epochs={r['epochs_run']}  {r['seconds']}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "channel_results.csv", index=False)
    summary = metrics.summarize(int(df.true_positives.sum()), int(df.false_positives.sum()), int(df.false_negatives.sum()))
    summary.update({"spacecraft": spacecraft, "channels": len(df), "config": cfg.to_dict()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=list))
    return df, summary
