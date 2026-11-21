"""As-flown EDL event times for Perseverance (Mars 2020) and Curiosity (MSL), with sources.

All times are converted to SPICE ephemeris time (TDB seconds past J2000) by `event_ets()`.
Each mission has the same six phases, so a model trained on one landing can be tested on the other:

    cruise -> approach -> entry -> parachute -> powered_descent -> sky_crane -> (touchdown)

Phase boundaries
    approach         start of EDL reconstruction (cruise stage already separated) .. entry interface
    entry            entry interface (radius 3522.2 km, ~125 km altitude) .. parachute deploy
    parachute        parachute deploy .. backshell separation  (includes heat-shield separation)
    powered_descent  backshell separation .. rover separation (sky-crane start)
    sky_crane        rover separation .. touchdown
"""
import spiceypy as sp

PHASES = ["cruise", "approach", "entry", "parachute", "powered_descent", "sky_crane"]

# ---------------------------------------------------------------------------
# Mars 2020 / Perseverance — 18 Feb 2021
#
# * Touchdown: official touchdown time from the NAIF EDL SPK comments
#   (m2020_edl_v01.bsp, landing-site segment start, 2021-FEB-18 20:44:58.003 TDB).
# * Entry, parachute, heat-shield and backshell times: Karlgaard et al., "Mars Entry, Descent,
#   and Landing Instrumentation 2: Trajectory, Aerodynamics, and Atmosphere Reconstruction"
#   (NTRS 20210024320), event table, seconds after on-board GN&C init. GN&C init is placed at
#   touchdown - 958.874 s; this agrees with the SPK start to 0.11 s and puts the vehicle at the
#   paper's entry-interface radius (3522.2 km) to within 10 m.
# * Rover separation: start of the "mated -> suspended" DIMU-to-rover offset in the SPK comments
#   (ET 666953081.6396100).
# ---------------------------------------------------------------------------
_M2020_TD = "2021-02-18 20:44:58.003 TDB"
_M2020_GNC_OFFSETS = {          # seconds after GN&C init (Karlgaard et al., Mars 2020)
    "entry_interface": 540.140,
    "parachute_deploy": 780.750,
    "heatshield_sep": 803.374,
    "backshell_sep": 897.671,
    "touchdown": 958.874,
}
_M2020_ROVER_SEP_ET = 666953081.6396100

# ---------------------------------------------------------------------------
# MSL / Curiosity — 6 Aug 2012
#
# * Entry, parachute, heat shield, rover separation, touchdown: NASA as-flown timeline
#   ("Timeline: Mission Milestones During Curiosity's Landing", nasa.gov, spacecraft event time,
#   converted from PDT to UTC). Touchdown agrees with the NAIF SPK official touchdown
#   (2012 AUG 06 05:19:04.475 TDB = 05:17:57.29 UTC) to 0.01 s.
# * Backshell separation is not in that table. It is computed from the as-flown GN&C mode
#   durations in Way et al., "Preliminary Assessment of the MSL EDL Simulation", AAS 13-420,
#   Table 5: heat-shield sep + modes 18..23 (3.016+2.109+13.125+62.5+14.875+1.422 = 97.047 s).
#   The same table reproduces NASA's heat-shield (+0.05 s), rover-sep (+0.02 s) and
#   touchdown (-0.06 s) times, so the derived value is accurate to ~0.1 s.
# ---------------------------------------------------------------------------
_MSL_UTC = {
    "entry_interface": "2012-08-06T05:10:45.7",
    "parachute_deploy": "2012-08-06T05:15:04.9",
    "heatshield_sep": "2012-08-06T05:15:24.6",
    "rover_sep": "2012-08-06T05:17:38.6",
    "touchdown": "2012-08-06T05:17:57.3",
}
_MSL_BACKSHELL_AFTER_HS = 3.015625 + 2.109375 + 13.125 + 62.5 + 14.875 + 1.421875


def event_ets(mission: str) -> dict[str, float]:
    """Event name -> ephemeris time. Requires the leap-seconds kernel to be loaded."""
    if mission == "m2020":
        t0 = sp.str2et(_M2020_TD) - _M2020_GNC_OFFSETS["touchdown"]
        ev = {k: t0 + v for k, v in _M2020_GNC_OFFSETS.items()}
        ev["rover_sep"] = _M2020_ROVER_SEP_ET
    elif mission == "msl":
        ev = {k: sp.str2et(v.replace("T", " ") + " UTC") for k, v in _MSL_UTC.items()}
        ev["backshell_sep"] = ev["heatshield_sep"] + _MSL_BACKSHELL_AFTER_HS
    else:
        raise ValueError(mission)
    return dict(sorted(ev.items(), key=lambda kv: kv[1]))


# event that *starts* each EDL phase (approach starts at the beginning of the EDL data)
PHASE_START_EVENT = {
    "entry": "entry_interface",
    "parachute": "parachute_deploy",
    "powered_descent": "backshell_sep",
    "sky_crane": "rover_sep",
}
