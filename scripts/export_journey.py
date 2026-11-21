"""Export the Mars 2020 (Perseverance) Earth -> Mars cruise for the 3D journey view.

Reads NASA/NAIF SPICE kernels already used by module 2 (data/raw/spice) and writes
models/serving/m2020_journey.json, served by the API at /api/journey.

* Frame: ECLIPJ2000, Sun-centred, so planetary orbits lie (almost) flat.
* Spacecraft: reconstructed cruise trajectory m2020_cruise_od138_v1.bsp (NAIF id -168),
  which starts about an hour after launch and ends at entry, descent and landing.
* Planets: DE438s ephemeris. Mars uses its barycentre (id 4); the offset from Mars'
  centre is under a metre, far below anything visible here.

    python scripts/export_journey.py
"""
import json
from pathlib import Path

import numpy as np
import spiceypy as sp

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "data" / "raw" / "spice"
OUT = ROOT / "models" / "serving" / "m2020_journey.json"
AU_KM = 149_597_870.7
C_KM_S = 299_792.458
SHIP = -168
STEP_S = 6 * 3600            # one sample every 6 hours
LAUNCH_UTC = "2020-07-30T11:50:00"     # Atlas V launch, Cape Canaveral
LANDING_UTC = "2021-02-18T20:55:00"    # touchdown, Jezero crater (Earth-received time ~20:55 UTC)
APPROACH_DAYS = 45                     # NASA's approach phase: last ~45 days before landing


def pos(target, ets):
    p, _ = sp.spkpos(target, ets, "ECLIPJ2000", "NONE", "SUN")
    return np.asarray(p)


def state(target, ets):
    s, _ = sp.spkezr(target, ets, "ECLIPJ2000", "NONE", "SUN")
    return np.asarray(s)


def main():
    for k in ("naif0012.tls", "de438s.bsp", "m2020_cruise_od138_v1.bsp"):
        sp.furnsh(str(KERNELS / k))
    cov = sp.spkcov(str(KERNELS / "m2020_cruise_od138_v1.bsp"), SHIP)
    start, end = sp.wnfetd(cov, 0)
    ets = np.append(np.arange(start, end, STEP_S), end)

    ship = state(str(SHIP), ets)
    earth = pos("EARTH", ets)
    mars = pos("MARS BARYCENTER", ets)
    r_ship = ship[:, :3]
    d_earth = np.linalg.norm(r_ship - earth, axis=1)
    d_mars = np.linalg.norm(r_ship - mars, axis=1)
    speed = np.linalg.norm(ship[:, 3:], axis=1)

    land_et = sp.str2et(LANDING_UTC)
    launch_et = sp.str2et(LAUNCH_UTC)
    phase = ["approach" if land_et - et <= APPROACH_DAYS * 86400 else "cruise" for et in ets]

    # Full orbit rings, one orbital period each, starting at launch
    ring = lambda body, days: (pos(body, launch_et + np.linspace(0, days * 86400, 241)) / AU_KM).round(5).tolist()

    au = lambda a: (a / AU_KM).round(5).tolist()
    out = {
        "source": "NASA/JPL NAIF SPICE: m2020_cruise_od138_v1.bsp (reconstructed cruise), de438s.bsp (planets)",
        "frame": "ECLIPJ2000, Sun-centred, units AU",
        "launch_utc": LAUNCH_UTC,
        "landing_utc": LANDING_UTC,
        "step_hours": STEP_S / 3600,
        "t_utc": [sp.et2utc(e, "ISOC", 0) for e in ets],
        "day": ((ets - launch_et) / 86400).round(3).tolist(),
        "phase": phase,
        "ship": au(r_ship),
        "earth": au(earth),
        "mars": au(mars),
        "speed_sun_km_s": speed.round(3).tolist(),
        "dist_earth_km": d_earth.round(0).tolist(),
        "dist_mars_km": d_mars.round(0).tolist(),
        "light_time_min": (d_earth / C_KM_S / 60).round(3).tolist(),
        "dist_sun_au": (np.linalg.norm(r_ship, axis=1) / AU_KM).round(5).tolist(),
        "earth_orbit": ring("EARTH", 365.25),
        "mars_orbit": ring("MARS BARYCENTER", 687.0),
        "total_path_km": float(np.sum(np.linalg.norm(np.diff(r_ship, axis=0), axis=1)).round(0)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{OUT.relative_to(ROOT)}: {len(ets)} samples, {out['t_utc'][0]} -> {out['t_utc'][-1]}, "
          f"path {out['total_path_km'] / 1e6:.0f} million km, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
