"""FastAPI application factory."""
from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from .config import get_config
from .db import init_db
from .routers import gemini3, twin, workspaces

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if get_config().is_dev else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Sovereign AI Workspace")
app.include_router(workspaces.router)
app.include_router(gemini3.router)
app.include_router(twin.router)
app.include_router(twin.router_v0)  # Support /api/twin for frontend compatibility


@app.on_event("startup")
async def startup() -> None:
    """Initialize database on application startup."""
    config = get_config()
    LOGGER.info("Starting Zylox backend (env=%s, db=%s)", config.APP_ENV, config.DATABASE_URL.split("://")[0])
    await init_db()
    LOGGER.info("Database initialized successfully")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint with basic system info."""
    config = get_config()
    db_type = config.DATABASE_URL.split("://")[0] if "://" in config.DATABASE_URL else "unknown"
    return {
        "status": "ok",
        "env": config.APP_ENV,
        "db_type": db_type,
    }

