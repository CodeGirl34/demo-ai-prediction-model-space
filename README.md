# Demo AI Prediction Model Space

**Phase-aware spacecraft anomaly detection** — an AI system that learns how a spacecraft
normally behaves, flags abnormal changes, and (next) understands which mission phase it is in:
Earth → Mars cruise or entry, descent & landing.

```
Mars 2020 trajectory ──► Phase classifier ──┐
   (+ injected faults)                      ├──► Phase-aware anomaly engine ──► FastAPI ──► Flutter app
MSL / SMAP telemetry ───────────────────────┘        (LSTM + dynamic threshold)
```

| Module | Data | Status |
|---|---|---|
| **1. Anomaly engine** | NASA SMAP & MSL telemetry with expert-labeled anomalies | ✅ implemented, validated |
| 2. Mission phase classifier | Mars 2020 (Perseverance) reconstructed trajectory from NASA SPICE | planned |
| 3. Synthetic fault injection | Faults inserted into the trajectory → labeled test data | planned |
| 4. API + mobile app | FastAPI backend, Flutter dashboard | planned |

---

## Module 1 — Anomaly engine

Re-implementation in modern PyTorch of the method from
Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
Thresholding*, KDD 2018 ([paper](https://arxiv.org/abs/1802.04431),
[reference code](https://github.com/khundman/telemanom)).

**How it works**

1. **Learn normal behaviour** — one LSTM per telemetry channel (2 layers × 80 units) reads the
   last 250 timesteps (telemetry value + 54 one-hot command flags) and predicts the next value.
2. **Prediction error** — where the spacecraft does something the model did not expect, the
   error grows. Errors are smoothed with an exponentially weighted moving average.
3. **Dynamic threshold** — no fixed cut-off. In a sliding window of recent errors the engine
   searches thresholds `mean + z·std` (z = 2.5 … 12) and picks the one that best separates a few
   compact anomalous regions from the rest. It also checks for unexpected *drops*.
4. **Pruning** — weak detections whose peak error is < 13 % above the next-highest are dropped,
   which removes most false alarms.

**Evaluation** (as in the paper): a labeled anomaly counts as found (TP) if any detected
sequence overlaps it; a detection overlapping no labeled anomaly is a false positive.

### Validation against the original implementation

The Kaggle archive ships the original 2018 LSTM predictions and run log. Feeding those
predictions into this repo's thresholding code (`scripts/validate_threshold.py`):

| | Channels identical to original | Precision | Recall | F0.5 |
|---|---|---|---|---|
| MSL | **27 / 27** | 0.962 | 0.694 | 0.893 |
| SMAP | 50 / 55 ¹ | 0.838 | 0.899 | 0.849 |

¹ The remaining SMAP differences come from the label file: the Kaggle labels were revised after
the 2018 run, so a few channels are being scored against different ground truth.

### Results — our own trained LSTMs on MSL

Full pipeline (training from scratch on a MacBook Air M3 CPU, ~27 min, single seed):

| | TP | FP | FN | Precision | Recall | F1 | F0.5 |
|---|---|---|---|---|---|---|---|
| **This repo (PyTorch, 1 run)** | 20 | 3 | 16 | **0.870** | 0.556 | 0.678 | 0.781 |
| Original 2018 Telemanom LSTMs ² | 25 | 1 | 11 | 0.962 | 0.694 | 0.806 | 0.893 |

² The original models' saved predictions, scored with the same code and the current labels.

19 of 27 channels give exactly the same result as the original. The 8 that differ
(M-2, M-3, P-15, C-2, F-4, D-15, D-16, F-8) differ in both directions. For example, we catch
M-2, which the original missed. This is expected run-to-run variance of per-channel LSTMs: the
original was trained once with Keras, and the random initialisation alone moves single channels.
Next steps to close the gap: train several seeds per channel and average the predictions.

![P-11: both labeled anomalies detected; the later error spike at ~3250 is correctly not flagged](results/msl/plots/P-11.png)

More plots: [M-6](results/msl/plots/M-6.png) · [F-7](results/msl/plots/F-7.png) · per-channel
numbers in [`results/msl/channel_results.csv`](results/msl/channel_results.csv).

---

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py                    # NASA SMAP/MSL from Kaggle (~86 MB)
python scripts/validate_threshold.py MSL           # check thresholding vs. original
python scripts/run_anomaly_engine.py --spacecraft MSL
python scripts/plot_channel.py M-6                 # results/msl/plots/M-6.png
```

## Project layout

```
src/spacecraft_ai/
  anomaly/          # module 1
    config.py       #   hyperparameters (defaults = original paper)
    data.py         #   load channels, build windows
    model.py        #   PyTorch LSTM forecaster + training w/ early stopping
    threshold.py    #   error smoothing, dynamic threshold, pruning
    metrics.py      #   sequence-level precision / recall / F-scores
    pipeline.py     #   train → predict → detect → evaluate, per channel
  phases/           # module 2 (planned)
scripts/            # command-line entry points
results/            # metrics, per-channel results, plots, trained models (.pt)
data/               # downloaded data (git-ignored)
```

## Data

* **SMAP / MSL telemetry** — [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl),
  originally released by NASA JPL with Hundman et al. 2018. MSL = Curiosity rover (27 channels),
  SMAP = Soil Moisture Active Passive satellite (55 channels). Values are scaled and channel
  names anonymized; anomalies were labeled by mission experts from incident reports.
* **Mars 2020 trajectory** (module 2) — reconstructed cruise and EDL ephemerides from
  [NASA NAIF SPICE](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/).
  These are navigation reconstructions, not raw sensor data, and contain no labeled anomalies —
  which is why module 3 injects documented synthetic faults for evaluation.

The two datasets come from different spacecraft and are never merged row-by-row; they share the
*anomaly engine* and the *app*, not the data.
