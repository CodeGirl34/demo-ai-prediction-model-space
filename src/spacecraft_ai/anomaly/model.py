"""Per-channel LSTM forecaster that learns 'normal' telemetry behaviour."""
import copy

import numpy as np
import torch
from torch import nn

from .config import Config


def pick_device() -> torch.device:
    # Apple-silicon GPU is often slower than CPU for small LSTMs; CPU is the safe default.
    return torch.device("cpu")


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, cfg: Config):
        super().__init__()
        h1, h2 = cfg.hidden_sizes
        self.lstm1 = nn.LSTM(n_features, h1, batch_first=True)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(h2, cfg.n_predictions)
        if cfg.keras_compat:
            self._keras_init()

    def _keras_init(self):
        """Match the Keras defaults used by the original Telemanom model.

        Glorot-uniform input weights, orthogonal recurrent weights, zero biases
        except the forget gate (= 1). The forget-gate bias makes it easy for the
        LSTM to carry the latest value forward, which matters when test
        telemetry moves outside the range seen in training.
        """
        for lstm in (self.lstm1, self.lstm2):
            h = lstm.hidden_size
            nn.init.xavier_uniform_(lstm.weight_ih_l0)
            nn.init.orthogonal_(lstm.weight_hh_l0)
            nn.init.zeros_(lstm.bias_ih_l0)
            nn.init.zeros_(lstm.bias_hh_l0)
            with torch.no_grad():
                lstm.bias_ih_l0[h:2 * h] = 1.0      # PyTorch gate order: input, forget, cell, output
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])       # last hidden state
        return self.head(x)


def train(X: np.ndarray, y: np.ndarray, cfg: Config, verbose: bool = False) -> tuple[LSTMForecaster, dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device()

    # Random validation split: the original shuffles the windows before Keras takes the last 20 %.
    n_val = max(1, int(len(X) * cfg.validation_split))
    perm = rng.permutation(len(X))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    model = LSTMForecaster(X.shape[2], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), cfg.batch_size):
            b = tr_idx[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b].to(device)), yt[b].to(device))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xt[val_idx].to(device)), yt[val_idx].to(device)).item()
        history.append(val_loss)
        if verbose:
            print(f"    epoch {epoch + 1:2d}  val_mse={val_loss:.5f}")

        if val_loss < best - cfg.min_delta:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if best_state is None:
                best_state = copy.deepcopy(model.state_dict())
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, {"epochs_run": len(history), "best_val_mse": best, "val_history": history}


@torch.no_grad()
def predict(model: LSTMForecaster, X: np.ndarray, batch: int = 512) -> np.ndarray:
    """One-step-ahead prediction for every window (Telemanom 'first' aggregation)."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.from_numpy(X[i:i + batch]).to(device))[:, 0].cpu().numpy())
    return np.concatenate(out)
