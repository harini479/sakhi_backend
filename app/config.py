"""Centralized environment-driven runtime configuration for Google Cloud Run and local dev."""

import os
from dataclasses import dataclass, field
from typing import List


def _parse_bool(raw: str, default: bool = False) -> bool:
    val = (raw or "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def _parse_origins(raw: str, is_production: bool) -> List[str]:
    origins: List[str] = []
    for part in (raw or "").split(","):
        cleaned = part.strip().rstrip("/")
        if not cleaned:
            continue
        # Never allow wildcard origins in production or when authenticated requests are used
        if is_production and cleaned == "*":
            continue
        origins.append(cleaned)
    return origins


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    log_level: str = "INFO"
    port: int = 8080
    google_cloud_project: str = ""
    firebase_project_id: str = ""
    firestore_database_id: str = "(default)"
    require_firebase_auth: bool = False
    cors_origins: List[str] = field(default_factory=list)
    gemini_model: str = "gemini-4-hydrogen"
    gemini_fallback_model: str = "gemini-3.1-flash-lite"
    gemini_timeout_seconds: float = 6.0
    openai_model: str = "gpt-4o-mini"
    hospital_search_radius_meters: int = 10000
    webhook_secret: str = ""

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


def load_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "development").strip().lower() or "development"
    is_prod = app_env == "production"

    # Support CORS_ORIGINS (primary) with backward-compatible ALLOWED_ORIGINS fallback
    raw_cors = os.getenv("CORS_ORIGINS", "").strip() or os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw_cors and not is_prod:
        raw_cors = "http://localhost:5173,http://localhost:3000"

    try:
        port = int(os.getenv("PORT", "8080"))
    except ValueError:
        port = 8080

    try:
        gemini_timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "6.0"))
    except ValueError:
        gemini_timeout = 6.0

    try:
        radius = int(os.getenv("HOSPITAL_SEARCH_RADIUS_METERS", "10000"))
    except ValueError:
        radius = 10000

    gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    firebase_project = os.getenv("FIREBASE_PROJECT_ID", "").strip() or gcp_project
    firestore_db = os.getenv("FIRESTORE_DATABASE_ID", "(default)").strip() or "(default)"

    return Settings(
        app_env=app_env,
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        port=port,
        google_cloud_project=gcp_project,
        firebase_project_id=firebase_project,
        firestore_database_id=firestore_db,
        require_firebase_auth=_parse_bool(
            os.getenv("REQUIRE_FIREBASE_AUTH", ""),
            default=is_prod,
        ),
        cors_origins=_parse_origins(raw_cors, is_production=is_prod),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-4-hydrogen").strip() or "gemini-4-hydrogen",
        gemini_fallback_model=(
            os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite").strip()
            or "gemini-3.1-flash-lite"
        ),
        gemini_timeout_seconds=max(2.0, min(gemini_timeout, 30.0)),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        hospital_search_radius_meters=max(1000, min(radius, 50000)),
        webhook_secret=(
            os.getenv("WEBHOOK_SECRET", "").strip()
            or os.getenv("INTERNAL_API_SECRET", "").strip()
        ),
    )


def get_settings() -> Settings:
    return load_settings()
