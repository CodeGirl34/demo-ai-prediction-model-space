"""FastAPI app.

    uvicorn spacecraft_ai.api.app:app --app-dir src --port 8000

Interactive docs at /docs. If the Flutter web build exists (app/build/web), it is served at /.
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..faults.inject import FAULT_TYPES
from ..faults.experiment import CHANNELS
from ..phases.classifier import EDL_PHASES
from .engine import Engine, ROOT, MISSIONS

FaultKind = Literal[tuple(FAULT_TYPES)]
Phase = Literal[tuple(EDL_PHASES)]
Channel = Literal[tuple(CHANNELS)]
MissionId = Literal[tuple(MISSIONS)]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = Engine()
    yield
    state.clear()


app = FastAPI(title="Spacecraft AI", version="0.1.0", lifespan=lifespan,
              description="Phase-aware spacecraft anomaly detection: Mars landing replays with fault "
                          "injection, mission-phase classification, and NASA MSL telemetry anomalies.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def engine() -> Engine:
    return state["engine"]


class FaultRequest(BaseModel):
    kind: FaultKind
    phase: Phase
    magnitude_frac: float | None = Field(None, ge=0, le=1, description="fault size as a fraction of the channel range")


class ReplayRequest(BaseModel):
    channel: Channel = "vertical_speed_m_s"
    faults: list[FaultRequest] = Field(default_factory=list, max_length=5)
    random_faults: int = Field(0, ge=0, le=4, description="extra faults with random type and phase")
    seed: int = 0


@app.get("/api/health")
def health():
    e = engine()
    return {"status": "ok", "models_trained_on": e.trained_on, "channels": CHANNELS,
            "fault_types": FAULT_TYPES, "phases": EDL_PHASES}


@app.get("/api/summary")
def summary():
    return engine().summary()


@app.get("/api/missions")
def missions():
    return engine().missions()


@app.get("/api/missions/{mission}/trajectory")
def trajectory(mission: MissionId, step: int = Query(10, ge=1, le=1000)):
    return engine().trajectory(mission, step)


@app.post("/api/missions/{mission}/replay")
def replay(mission: MissionId, req: ReplayRequest):
    return engine().replay(mission, req.channel, [f.model_dump() for f in req.faults], req.random_faults, req.seed)


@app.get("/api/journey")
def journey():
    """Perseverance's Earth -> Mars cruise from NASA SPICE (see scripts/export_journey.py)."""
    path = ROOT / "models" / "serving" / "m2020_journey.json"
    if not path.exists():
        raise HTTPException(503, "journey data missing: run scripts/export_journey.py")
    return Response(path.read_bytes(), media_type="application/json")


@app.get("/api/telemetry/channels")
def telemetry_channels():
    return engine().telemetry_channels()


@app.get("/api/telemetry/channels/{chan_id}")
def telemetry_channel(chan_id: str):
    try:
        return engine().telemetry_channel(chan_id)
    except KeyError:
        raise HTTPException(404, f"unknown channel {chan_id}")
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


WEB = ROOT / "app" / "build" / "web"
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
