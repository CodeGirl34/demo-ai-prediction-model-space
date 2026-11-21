"""Download NASA SPICE kernels and build the Mars landing / cruise CSVs.

    python scripts/build_mars_dataset.py              # EDL at 10 Hz, cruise every 10 min

Outputs (data/processed/mars_edl/):
    m2020_cruise.csv   Perseverance Earth -> Mars cruise
    m2020_edl.csv      Perseverance entry, descent & landing, labeled phases
    msl_edl.csv        Curiosity entry, descent & landing, labeled phases
    events.csv         as-flown event times used for the labels
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import numpy as np
import pandas as pd
import spiceypy as sp

from spacecraft_ai.phases import extract
from spacecraft_ai.phases.events import event_ets

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "mars_edl"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edl-dt", type=float, default=0.1, help="EDL sample interval, s")
    ap.add_argument("--cruise-dt", type=float, default=600.0, help="cruise sample interval, s")
    a = ap.parse_args()

    extract.download_kernels()
    extract.load_kernels()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for mission in extract.MISSIONS:
        df = extract.edl_table(mission, a.edl_dt)
        df.to_csv(OUT / f"{mission}_edl.csv", index=False)
        print(f"{mission}_edl.csv     {len(df):6d} rows   phases: {df.phase.value_counts().to_dict()}")
        for name, et in event_ets(mission).items():
            i = int(np.argmin(np.abs(df.et_tdb_s.to_numpy() - et)))
            rows.append({"mission": mission, "event": name, "utc": sp.et2utc(et, "ISOC", 3), "et_tdb_s": et,
                         "altitude_km": round(df.altitude_km.iat[i], 3),
                         "speed_rel_mars_m_s": round(df.speed_rel_mars_m_s.iat[i], 1)})

    cr = extract.cruise_table(a.cruise_dt)
    cr.to_csv(OUT / "m2020_cruise.csv", index=False)
    print(f"m2020_cruise.csv  {len(cr):6d} rows   {cr.timestamp_utc.iat[0]} .. {cr.timestamp_utc.iat[-1]}")

    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "events.csv", index=False)
    print(ev.to_string(index=False))
