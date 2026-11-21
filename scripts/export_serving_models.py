"""Train the models the API serves and save them to models/serving/.

Trained on Perseverance (Mars 2020), exactly as in module 3, so replays on Curiosity are a fair
cross-mission test. Also checks that the clean-landing false alarms match the committed module-3
results (proves the served models are the measured ones).

    python scripts/build_mars_dataset.py      # first
    python scripts/export_serving_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import joblib
import pandas as pd
import torch

from spacecraft_ai.faults.experiment import Experiment, CFG, CHANNELS, CONTEXT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "mars_edl"
OUT = ROOT / "models" / "serving"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = (pd.read_csv(DATA / f"{m}_edl.csv") for m in ("m2020", "msl"))
    exp = Experiment(train, test, seed=0, threads=4)

    joblib.dump(exp.phase_model, OUT / "phase_classifier.joblib")
    n_features = {}
    for (ch, variant), net in exp.models.items():
        name = f"{variant}__{ch}.pt"
        torch.save(net.state_dict(), OUT / name)
        n_features[name] = int(net.lstm1.input_size)
    meta = {
        "trained_on": "m2020",
        "channels": CHANNELS,
        "context": CONTEXT,
        "scaler": {"lo": exp.scaler.lo.to_dict(), "hi": exp.scaler.hi.to_dict()},
        "n_features": n_features,
        "config": CFG.to_dict(),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", sorted(p.name for p in OUT.iterdir()))

    # consistency check against committed module-3 results (same seed, same direction)
    fa = pd.DataFrame(exp.false_alarms_on_clean())
    ref = pd.read_csv(ROOT / "results" / "faults" / "false_alarms_clean.csv")
    ref = ref[ref.direction == "Perseverance -> Curiosity"].drop(columns="direction")
    key = ["channel", "detector", "start", "end", "phase"]
    same = fa[key].sort_values(key).reset_index(drop=True).equals(ref[key].sort_values(key).reset_index(drop=True))
    print("clean-landing alarms identical to module-3 results:", same)
    if not same:
        print(fa.to_string()); print(ref.to_string())
        sys.exit(1)
