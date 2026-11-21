"""Hyperparameters. Defaults follow the original Telemanom config.yaml (Hundman et al., KDD 2018)."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- windows ---
    l_s: int = 250              # input window length (timesteps of history)
    n_predictions: int = 10     # timesteps predicted ahead per window

    # --- LSTM ---
    hidden_sizes: tuple = (80, 80)
    dropout: float = 0.3
    epochs: int = 35
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    patience: int = 10
    min_delta: float = 3e-4

    # --- error smoothing / dynamic threshold ---
    error_batch_size: int = 70      # timesteps per error batch
    window_size: int = 30           # number of error batches in a threshold window
    smoothing_perc: float = 0.05    # EWMA span = error_batch_size * window_size * smoothing_perc
    error_buffer: int = 100         # timesteps added around each anomaly
    p: float = 0.13                 # minimum percent decrease between ranked anomalies (pruning)
    sd_lim: float = 12.0            # max z for threshold search

    seed: int = 42

    def to_dict(self):
        return asdict(self)
