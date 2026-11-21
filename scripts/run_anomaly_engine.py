"""Train and evaluate the anomaly engine on NASA SMAP/MSL telemetry.

    python scripts/run_anomaly_engine.py --spacecraft MSL
    python scripts/run_anomaly_engine.py --spacecraft MSL --channels M-6 T-4 --epochs 5   # quick test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pathlib import Path

from spacecraft_ai.anomaly.config import Config
from spacecraft_ai.anomaly.data import DEFAULT_ROOT
from spacecraft_ai.anomaly.pipeline import run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacecraft", choices=["MSL", "SMAP"], default="MSL")
    ap.add_argument("--channels", nargs="*")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    if a.epochs:
        cfg.epochs = a.epochs
    out = a.out or ROOT / "results" / a.spacecraft.lower()
    _, summary = run(a.spacecraft, cfg, DEFAULT_ROOT, out, a.channels, a.verbose)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
