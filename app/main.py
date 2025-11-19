"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI

from .db import init_db
from .routers import workspaces

app = FastAPI(title="Sovereign AI Workspace")
app.include_router(workspaces.router)


@app.on_event("startup")
async def startup() -> None:
    await init_db()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

