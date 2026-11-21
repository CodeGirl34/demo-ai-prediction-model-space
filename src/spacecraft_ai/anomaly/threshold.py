"""Non-parametric dynamic thresholding of LSTM prediction errors.

Re-implementation of the method in Hundman et al., "Detecting Spacecraft
Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (KDD 2018),
following the logic of the reference code (github.com/khundman/telemanom,
telemanom/errors.py), rewritten without the more_itertools dependency.

Pipeline:
  1. raw error  e_t = |y_t - y_hat_t|
  2. smooth with an EWMA                  -> e_s
  3. slide a window of past errors; in each window pick the threshold
     epsilon = mean + z*sd (z in 2.5..12) that best separates anomalies
     (both high errors and "inverted" low errors)
  4. prune weak anomalies whose peak error is < p (13%) above the next one
  5. merge flagged indices into (start, end) anomaly sequences
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config


def consecutive_groups(idx: np.ndarray) -> list[np.ndarray]:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    return np.split(idx, breaks)


def to_sequences(idx: np.ndarray) -> list[tuple[int, int]]:
    return [(int(g[0]), int(g[-1])) for g in consecutive_groups(idx) if g[0] != g[-1]]


def _add_buffer(i_anom: np.ndarray, buffer: np.ndarray, n: int) -> np.ndarray:
    if i_anom.size == 0:
        return i_anom.astype(int)
    out = np.concatenate([i_anom, (i_anom[:, None] + buffer).ravel(), (i_anom[:, None] - buffer).ravel()])
    out = out[(out >= 0) & (out < n)]
    return np.unique(out).astype(int)


def smooth_errors(y: np.ndarray, y_hat: np.ndarray, cfg: Config, fill_initial: bool = True) -> np.ndarray:
    e = np.abs(y - y_hat)
    span = int(cfg.error_batch_size * cfg.window_size * cfg.smoothing_perc)
    e_s = pd.Series(e).ewm(span=span).mean().to_numpy().copy()
    # first l_s values have little history: replace with an average
    # (the reference code skips this for channel C-2, whose anomaly starts early)
    if fill_initial:
        e_s[:cfg.l_s] = np.mean(e_s[:cfg.l_s * 2])
    return e_s


class _Window:
    def __init__(self, e_s_all, y_all, start, end, window_num, n_total, cfg: Config):
        self.cfg = cfg
        self.window_num = window_num
        self.e_s = e_s_all[start:end]
        self.mean, self.sd = float(np.mean(self.e_s)), float(np.std(self.e_s))
        self.e_s_inv = self.mean + (self.mean - self.e_s)
        y = y_all[start:end]
        self.sd_values = float(np.std(y))
        hi, lo = np.percentile(y, [95, 5])
        self.inter_range = float(hi - lo)
        self.epsilon = {False: self.mean + cfg.sd_lim * self.sd, True: self.mean + cfg.sd_lim * self.sd}
        self.i_anom = {False: np.array([], int), True: np.array([], int)}
        self.E_seq = {False: [], True: []}
        self.non_anom_max = {False: -1e6, True: -1e6}
        self.num_to_ignore = cfg.l_s * 2
        if n_total < 2500:
            self.num_to_ignore = cfg.l_s
        if n_total < 1800:
            self.num_to_ignore = 0

    def _errors(self, inverse):
        return self.e_s_inv if inverse else self.e_s

    def find_epsilon(self, inverse=False):
        e_s = self._errors(inverse)
        buffer = np.arange(1, self.cfg.error_buffer)
        max_score = -1e7
        for z in np.arange(2.5, self.cfg.sd_lim, 0.5):
            eps = self.mean + self.sd * z
            pruned = e_s[e_s < eps]
            i_anom = _add_buffer(np.argwhere(e_s >= eps).ravel(), buffer, len(e_s))
            if i_anom.size == 0 or pruned.size == 0:
                continue
            E_seq = to_sequences(i_anom)
            mean_dec = (self.mean - np.mean(pruned)) / self.mean
            sd_dec = (self.sd - np.std(pruned)) / self.sd
            score = (mean_dec + sd_dec) / (len(E_seq) ** 2 + len(i_anom))
            if score >= max_score and len(E_seq) <= 5 and len(i_anom) < len(e_s) * 0.5:
                max_score = score
                self.epsilon[inverse] = eps

    def compare_to_epsilon(self, already_flagged: np.ndarray, inverse=False):
        e_s = self._errors(inverse)
        eps = self.epsilon[inverse]
        # errors too small relative to the signal -> nothing to flag
        if not (self.sd > 0.05 * self.sd_values or self.e_s.max() > 0.05 * self.inter_range) \
                or not self.e_s.max() > 0.05:
            return
        i_anom = np.argwhere((e_s >= eps) & (e_s > 0.05 * self.inter_range)).ravel()
        if i_anom.size == 0:
            return
        i_anom = _add_buffer(i_anom, np.arange(1, self.cfg.error_buffer + 1), len(e_s))
        if self.window_num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:   # only the newest batch of the window is evaluated
            i_anom = i_anom[i_anom >= len(e_s) - self.cfg.error_batch_size]
        i_anom = np.unique(i_anom)

        pos = self.window_num * self.cfg.error_batch_size
        candidates = np.setdiff1d(np.arange(len(e_s)) + pos, np.append(already_flagged, i_anom + pos)) - pos
        self.non_anom_max[inverse] = float(np.max(e_s[candidates])) if candidates.size else -1e6
        self.i_anom[inverse] = i_anom
        self.E_seq[inverse] = to_sequences(i_anom)

    def prune(self, inverse=False):
        E_seq, e_s = self.E_seq[inverse], self._errors(inverse)
        if len(E_seq) == 0:
            return
        seq_max = np.array([e_s[s:e + 1].max() for s, e in E_seq])
        ranked = np.append(np.sort(seq_max)[::-1], self.non_anom_max[inverse])
        to_remove = []
        for i in range(len(ranked) - 1):
            if (ranked[i] - ranked[i + 1]) / ranked[i] < self.cfg.p:
                to_remove.extend(np.argwhere(seq_max == ranked[i]).ravel().tolist())
            else:
                to_remove = []
        keep = [seq for j, seq in enumerate(E_seq) if j not in set(to_remove)]
        if not keep:
            self.i_anom[inverse] = np.array([], int)
            return
        keep_idx = np.concatenate([np.arange(s, e + 1) for s, e in keep])
        self.i_anom[inverse] = self.i_anom[inverse][np.isin(self.i_anom[inverse], keep_idx)]

    def scores(self, idx: np.ndarray, start: int) -> list[dict]:
        out = []
        denom = self.mean + self.sd
        for g in consecutive_groups(idx):
            s = np.max(np.abs(self.e_s[g] - self.epsilon[False])) / denom
            s_inv = np.max(np.abs(self.e_s_inv[g] - self.epsilon[True])) / denom
            out.append({"start_idx": int(g[0] + start), "end_idx": int(g[-1] + start), "score": float(max(s, s_inv))})
        return out


@dataclass
class DetectionResult:
    e_s: np.ndarray
    sequences: list = field(default_factory=list)   # (start, end) in original test-array indices
    scores: list = field(default_factory=list)


def detect(y: np.ndarray, y_hat: np.ndarray, cfg: Config, fill_initial: bool = True) -> DetectionResult:
    """Run dynamic thresholding. `y`/`y_hat` are aligned so y[i] = test[i + l_s, 0]."""
    e_s = smooth_errors(y, y_hat, cfg, fill_initial)
    n = len(y)
    bs = cfg.error_batch_size
    window_size = cfg.window_size
    n_windows = int((n - bs * window_size) / bs)
    while n_windows < 0 and window_size > 1:
        window_size -= 1
        n_windows = int((n - bs * window_size) / bs)
    if n_windows < 0:
        raise ValueError(f"error_batch_size {bs} larger than series of length {n}")

    flagged = np.array([], int)
    scores = []
    for i in range(n_windows + 1):
        start = i * bs
        end = n if i == n_windows else window_size * bs + i * bs
        w = _Window(e_s, y, start, end, i, n, cfg)
        for inv in (False, True):
            w.find_epsilon(inv)
        for inv in (False, True):
            w.compare_to_epsilon(flagged, inv)
        if w.i_anom[False].size == 0 and w.i_anom[True].size == 0:
            continue
        for inv in (False, True):
            w.prune(inv)
        idx = np.unique(np.append(w.i_anom[False], w.i_anom[True])).astype(int)
        if idx.size == 0:
            continue
        scores += w.scores(idx, start)
        flagged = np.append(flagged, idx + start)

    seqs = [(s + cfg.l_s, e + cfg.l_s) for s, e in to_sequences(np.unique(flagged))]
    return DetectionResult(e_s=e_s, sequences=seqs, scores=scores)
