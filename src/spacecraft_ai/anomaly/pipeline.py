"""End-to-end: train LSTM(s) per channel, predict the test set, detect and score anomalies."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, model, threshold, metrics
from .config import Config


def run_channel(chan_id: str, true_seqs: list, cfg: Config, root: Path, model_dir: Path | None,
                verbose=False, threads: int | None = None) -> dict:
    if threads:
        torch.set_num_threads(threads)
    t0 = time.time()
    train_arr, test_arr = data.load_channel(chan_id, root)
    X_tr, y_tr = data.make_windows(train_arr, cfg.l_s, cfg.n_predictions)
    X_te, y_te = data.make_windows(test_arr, cfg.l_s, cfg.n_predictions)

    # Ensemble: same architecture, different random initialisation / shuffling; average predictions.
    preds, epochs = [], []
    for k in range(cfg.n_seeds):
        net, info = model.train(X_tr, y_tr, replace(cfg, seed=cfg.seed + k), verbose=verbose)
        if model_dir:
            model_dir.mkdir(parents=True, exist_ok=True)
            name = f"{chan_id}.pt" if cfg.n_seeds == 1 else f"{chan_id}_seed{k}.pt"
            torch.save(net.state_dict(), model_dir / name)
        preds.append(model.predict(net, X_te))
        epochs.append(info["epochs_run"])
    y_hat = np.mean(preds, axis=0)

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
        "epochs_run": epochs,
        "seconds": round(time.time() - t0, 1),
        "_y": y, "_y_hat": y_hat, "_e_s": det.e_s,
    }


def run(spacecraft: str, cfg: Config, root: Path, out_dir: Path, channels: list[str] | None = None,
        verbose=False, workers: int = 1):
    labels = data.load_labels(root, spacecraft)
    if channels:
        labels = labels[labels.chan_id.isin(channels)]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    model_dir = out_dir / "models"

    def _record(r, k):
        np.savez_compressed(pred_dir / f"{r['chan_id']}.npz", y=r.pop("_y"), y_hat=r.pop("_y_hat"), e_s=r.pop("_e_s"))
        print(f"[{k:2d}/{len(labels)}] {r['chan_id']:6s} TP={r['true_positives']} FP={r['false_positives']} "
              f"FN={r['false_negatives']}  epochs={r['epochs_run']}  {r['seconds']}s", flush=True)
        return r

    rows = []
    if workers <= 1:
        for k, row in enumerate(labels.itertuples(), 1):
            rows.append(_record(run_channel(row.chan_id, row.anomaly_sequences, cfg, root, model_dir, verbose), k))
    else:
        threads = max(1, (os.cpu_count() or 4) // workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_channel, row.chan_id, row.anomaly_sequences, cfg, root, model_dir, False, threads)
                    for row in labels.itertuples()]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(_record(f.result(), k))

    order = {c: i for i, c in enumerate(labels.chan_id)}
    df = pd.DataFrame(rows).sort_values("chan_id", key=lambda s: s.map(order))
    df.to_csv(out_dir / "channel_results.csv", index=False)
    summary = metrics.summarize(int(df.true_positives.sum()), int(df.false_positives.sum()), int(df.false_negatives.sum()))
    summary.update({"spacecraft": spacecraft, "channels": len(df), "config": cfg.to_dict()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=list))
    return df, summary
