"""Turn NASA SPICE trajectory kernels into tidy time-series tables.

What the numbers are (and are not):
  * The EDL trajectories are NASA's final reconstructions, made by integrating the Descent IMU
    (DIMU-A) measurements backwards from the known touchdown point. The table samples that
    reconstruction; the sample rate is our choice, not the IMU's.
  * `sensed_accel_m_s2` = d(v)/dt in the inertial J2000 frame minus point-mass Mars gravity,
    i.e. the non-gravitational acceleration the IMU felt (drag, parachute, engine thrust).
  * `altitude_km` is height above the official landing-site radius (not above local terrain
    under the vehicle, which differs by up to a few km far from the site).
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy as sp

from .events import event_ets, PHASE_START_EVENT

ROOT = Path(__file__).resolve().parents[3]
KERNEL_DIR = ROOT / "data" / "raw" / "spice"

_M20 = "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/spice_kernels"
_MSL = "https://naif.jpl.nasa.gov/pub/naif"
KERNELS = {
    "naif0012.tls": f"{_M20}/lsk/naif0012.tls",
    "pck00010.tpc": f"{_M20}/pck/pck00010.tpc",
    "de438s.bsp": f"{_M20}/spk/de438s.bsp",
    "m2020_cruise_od138_v1.bsp": f"{_M20}/spk/m2020_cruise_od138_v1.bsp",
    "m2020_edl_v01.bsp": f"{_M20}/spk/m2020_edl_v01.bsp",
    "m2020_ls_ops210303_iau2000_v1.bsp": f"{_M20}/spk/m2020_ls_ops210303_iau2000_v1.bsp",
    "msl_edl_v01.bsp": f"{_MSL}/pds/data/msl-m-spice-6-v1.0/mslsp_1000/data/spk/msl_edl_v01.bsp",
    "msl_ls_gc120806_iau2000_v3.bsp": f"{_MSL}/MSL/kernels/spk/msl_ls_gc120806_iau2000_v3.bsp",
}

MISSIONS = {
    #          DIMU-A id, landing-site id, EDL kernel
    "m2020": dict(dimu=-168031, site=-168900, edl="m2020_edl_v01.bsp", name="Mars 2020 / Perseverance"),
    "msl":   dict(dimu=-76031,  site=-76900,  edl="msl_edl_v01.bsp",   name="MSL / Curiosity"),
}
GM_MARS = 42828.37          # km^3/s^2


def download_kernels(dest: Path = KERNEL_DIR):
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in KERNELS.items():
        if not (dest / name).exists():
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, dest / name)


def load_kernels(dest: Path = KERNEL_DIR):
    sp.kclear()
    sp.furnsh([str(dest / k) for k in KERNELS])


def _coverage(kernel: str, obj: int) -> tuple[float, float]:
    cov = sp.spkcov(str(KERNEL_DIR / kernel), obj, sp.cell_double(100))
    return sp.wnfetd(cov, 0)


def _states(target: str, ets: np.ndarray, frame: str, observer: str) -> np.ndarray:
    return np.array([sp.spkezr(target, float(t), frame, "NONE", observer)[0] for t in ets])


def edl_table(mission: str, dt: float = 0.1) -> pd.DataFrame:
    """Sampled EDL trajectory (EDL start .. touchdown) with phase labels."""
    m = MISSIONS[mission]
    ev = event_ets(mission)
    start, _ = _coverage(m["edl"], m["dimu"])
    ets = np.arange(start, ev["touchdown"], dt)

    dimu = str(m["dimu"])
    inert = _states(dimu, ets, "J2000", "MARS")          # km, km/s
    fixed = _states(dimu, ets, "IAU_MARS", "MARS")       # Mars-fixed: planet-relative velocity
    site = np.linalg.norm(sp.spkpos(str(m["site"]), ev["touchdown"], "IAU_MARS", "NONE", "MARS")[0])

    r_vec, v_fix = fixed[:, :3], fixed[:, 3:]
    r = np.linalg.norm(r_vec, axis=1)
    speed = np.linalg.norm(v_fix, axis=1)
    v_vert = np.einsum("ij,ij->i", r_vec, v_fix) / r
    fpa = np.degrees(np.arcsin(np.clip(v_vert / np.maximum(speed, 1e-9), -1, 1)))
    lat = np.degrees(np.arcsin(r_vec[:, 2] / r))
    lon = np.degrees(np.arctan2(r_vec[:, 1], r_vec[:, 0]))

    # non-gravitational (sensed) acceleration in the inertial frame
    a_tot = np.gradient(inert[:, 3:], ets, axis=0)                       # km/s^2
    r_in = inert[:, :3]
    g = -GM_MARS * r_in / np.linalg.norm(r_in, axis=1)[:, None] ** 3
    sensed = np.linalg.norm(a_tot - g, axis=1) * 1000                    # m/s^2

    # phase labels from the as-flown event times
    phase = np.full(len(ets), "approach", dtype=object)
    for ph, evname in PHASE_START_EVENT.items():
        phase[ets >= ev[evname]] = ph

    return pd.DataFrame({
        "mission": mission,
        "timestamp_utc": [sp.et2utc(float(t), "ISOC", 3) for t in ets],
        "et_tdb_s": ets,
        "t_minus_touchdown_s": ets - ev["touchdown"],
        "x_mars_j2000_km": r_in[:, 0], "y_mars_j2000_km": r_in[:, 1], "z_mars_j2000_km": r_in[:, 2],
        "vx_mars_j2000_km_s": inert[:, 3], "vy_mars_j2000_km_s": inert[:, 4], "vz_mars_j2000_km_s": inert[:, 5],
        "latitude_deg": lat, "longitude_deg": lon,
        "radius_km": r,
        "altitude_km": r - site,
        "speed_rel_mars_m_s": speed * 1000,
        "vertical_speed_m_s": v_vert * 1000,
        "flight_path_angle_deg": fpa,
        "sensed_accel_m_s2": sensed,
        "phase": phase,
    })


def cruise_table(dt: float = 600.0) -> pd.DataFrame:
    """Mars 2020 interplanetary cruise (launch + ~1 h .. EDL start), Sun/Earth/Mars-relative."""
    start, end = _coverage("m2020_cruise_od138_v1.bsp", -168)
    # stop before the EDL kernel (Mars-centred, higher priority) takes over the same object
    end = min(end, _coverage("m2020_edl_v01.bsp", -168)[0]) - 1.0
    ets = np.append(np.arange(start, end, dt), end)
    sun = _states("-168", ets, "J2000", "SUN")
    earth = _states("-168", ets, "J2000", "EARTH")
    # Mars barycenter (4): de438s has no Mars-centre (499) chain for cruise; offset is < 1 m.
    mars = _states("-168", ets, "J2000", "MARS BARYCENTER")
    n = lambda a: np.linalg.norm(a, axis=1)
    return pd.DataFrame({
        "mission": "m2020",
        "timestamp_utc": [sp.et2utc(float(t), "ISOC", 3) for t in ets],
        "et_tdb_s": ets,
        "days_since_cruise_start": (ets - start) / 86400,
        "x_sun_j2000_km": sun[:, 0], "y_sun_j2000_km": sun[:, 1], "z_sun_j2000_km": sun[:, 2],
        "vx_sun_j2000_km_s": sun[:, 3], "vy_sun_j2000_km_s": sun[:, 4], "vz_sun_j2000_km_s": sun[:, 5],
        "distance_sun_km": n(sun[:, :3]),
        "speed_sun_km_s": n(sun[:, 3:]),
        "distance_earth_km": n(earth[:, :3]),
        "speed_rel_earth_km_s": n(earth[:, 3:]),
        "distance_mars_km": n(mars[:, :3]),
        "speed_rel_mars_km_s": n(mars[:, 3:]),
        "phase": "cruise",
    })
