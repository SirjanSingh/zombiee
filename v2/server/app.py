"""SurviveCity v2 FastAPI server — OpenEnv-compliant HTTP wrapper.

Endpoints:
    POST /reset    -> {seed?: int}              -> observation
    POST /step     -> action dict               -> observation
    GET  /state                                 -> state dict
    GET  /health                                -> "ok"

A single in-process SurviveCityV2Env is held module-level. For multi-tenant
hosting, run multiple uvicorn workers, each owning their own instance.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from survivecity_v2_env.env import SurviveCityV2Env

logger = logging.getLogger("survivecity_v2.server")

app = FastAPI(title="SurviveCity v2", version="2.0.0")
_env = SurviveCityV2Env()


class ResetReq(BaseModel):
    seed: Optional[int] = None


@app.post("/reset")
def reset(req: ResetReq) -> dict[str, Any]:
    return _env.reset(seed=req.seed)


@app.post("/step")
def step(action: dict[str, Any]) -> dict[str, Any]:
    return _env.step(action)


@app.get("/state")
def state() -> dict[str, Any]:
    return _env.state


@app.get("/health")
def health() -> str:
    return "ok"
