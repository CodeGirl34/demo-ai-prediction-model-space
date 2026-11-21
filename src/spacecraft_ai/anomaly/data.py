"""Loading SMAP/MSL telemetry channels and building LSTM training windows."""
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "smap_msl"


def _data_dir(root: Path) -> Path:
    # The Kaggle archive nests the arrays as data/data/{train,test}
    for cand in (root / "data" / "data", root / "data", root):
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    raise FileNotFoundError(f"Could not find train/ and test/ under {root}. Run scripts/download_data.py first.")


def load_labels(root: Path = DEFAULT_ROOT, spacecraft: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(root / "labeled_anomalies.csv")
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(literal_eval)
    if spacecraft:
        df = df[df["spacecraft"] == spacecraft]
    return df.reset_index(drop=True)


def load_channel(chan_id: str, root: Path = DEFAULT_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test) arrays of shape (timesteps, features).

    Column 0 is the (scaled) telemetry value; the remaining columns are
    one-hot encoded commands sent to the spacecraft.
    """
    d = _data_dir(root)
    return (np.load(d / "train" / f"{chan_id}.npy").astype(np.float32),
            np.load(d / "test" / f"{chan_id}.npy").astype(np.float32))


def make_windows(arr: np.ndarray, l_s: int, n_predictions: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide over `arr` producing X (n, l_s, features) and y (n, n_predictions).

    y holds the next `n_predictions` values of the telemetry channel (column 0).
    """
    total = l_s + n_predictions
    n = len(arr) - total + 1
    if n <= 0:
        raise ValueError(f"Sequence of length {len(arr)} is shorter than window {total}")
    idx = np.arange(total)[None, :] + np.arange(n)[:, None]
    win = arr[idx]                       # (n, total, features)
    X = win[:, :l_s, :]
    y = win[:, l_s:, 0]
    return X, y
