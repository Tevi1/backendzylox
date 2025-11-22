"""Centralized configuration for Zylox backend."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration from environment variables."""

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data.db"

    # Gemini API
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""  # Fallback for legacy compatibility
    
    # Model selection
    RAG_MODEL: str = "gemini-2.5-pro"  # Model for RAG queries (can be gemini-3-pro-preview if supported)
    CLASSIFICATION_MODEL: str = "gemini-2.0-flash-thinking-exp-01-21"  # Model for document classification

    # Encryption
    FILE_ENC_KEY_B64: str = ""

    # CORS
    CORS_ORIGINS: List[str] = ["*"]  # TODO: restrict in production

    # Environment
    APP_ENV: str = "dev"
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def get_gemini_api_key(self) -> str:
        """Get Gemini API key with fallback to GOOGLE_API_KEY."""
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY

    def validate_critical(self) -> None:
        """Validate that critical environment variables are set."""
        errors: List[str] = []
        if not self.get_gemini_api_key():
            errors.append("GEMINI_API_KEY (or GOOGLE_API_KEY) is required")
        if not self.FILE_ENC_KEY_B64:
            errors.append("FILE_ENC_KEY_B64 is required for file encryption")
        if errors:
            raise RuntimeError("Missing required environment variables: " + "; ".join(errors))

    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.APP_ENV.lower() == "dev"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.APP_ENV.lower() == "production"


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get cached configuration instance."""
    config = Config()
    config.validate_critical()
    return config

