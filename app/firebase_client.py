"""Cloud Run–compatible Firebase Admin & Firestore initialization.

Credential resolution order:
  1. FIREBASE_SERVICE_ACCOUNT_KEY env var → path to a service-account JSON file
     (use this for local development when ADC is not configured)
  2. Application Default Credentials (ADC) → automatic on Cloud Run via IAM,
     or locally after `gcloud auth application-default login`
"""

import logging
import os
from typing import Any, Dict, Optional

from .config import get_settings

logger = logging.getLogger("sakhi.firebase")

try:
    import firebase_admin  # type: ignore
    from firebase_admin import auth as firebase_auth  # type: ignore
    from firebase_admin import credentials as firebase_credentials  # type: ignore
    from firebase_admin import firestore as firebase_firestore  # type: ignore
except ImportError:
    firebase_admin = None  # type: ignore
    firebase_auth = None  # type: ignore
    firebase_credentials = None  # type: ignore
    firebase_firestore = None  # type: ignore

_firebase_app: Any = None
_firestore_client: Any = None
_init_attempted: bool = False
_init_error: Optional[str] = None


def _build_credential() -> Any:
    """
    Return the best available Firebase credential.

    Priority:
      1. FIREBASE_SERVICE_ACCOUNT_KEY   → path to service-account JSON (local dev)
      2. FIREBASE_SERVICE_ACCOUNT_JSON  → raw JSON string or base64 (cloud env vars)
      3. ApplicationDefault()           → Cloud Run IAM / gcloud ADC
    """
    import json
    import base64

    key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY", "").strip()
    if key_path and os.path.isfile(key_path):
        logger.info(
            "Using Firebase service account key file: %s", key_path,
            extra={"event": "firebase_cred_service_account"},
        )
        return firebase_credentials.Certificate(key_path)

    raw_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw_json:
        try:
            if raw_json.startswith("{"):
                cred_dict = json.loads(raw_json)
            else:
                cred_dict = json.loads(base64.b64decode(raw_json).decode("utf-8"))
            logger.info(
                "Using Firebase service account JSON from environment variable.",
                extra={"event": "firebase_cred_env_json"},
            )
            return firebase_credentials.Certificate(cred_dict)
        except Exception as exc:
            logger.warning(
                "Could not parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", exc,
                extra={"event": "firebase_cred_parse_error"},
            )

    # ApplicationDefault() automatically honours Cloud Run IAM and GOOGLE_APPLICATION_CREDENTIALS
    logger.info(
        "Using Application Default Credentials (ADC).",
        extra={"event": "firebase_cred_adc"},
    )
    return firebase_credentials.ApplicationDefault()


def init_firebase() -> bool:
    """Initialize Firebase Admin SDK.

    Uses a service-account JSON key for local dev (FIREBASE_SERVICE_ACCOUNT_KEY),
    or ADC on Cloud Run. Never hard-codes credentials.
    """
    global _firebase_app, _firestore_client, _init_attempted, _init_error

    if _init_attempted:
        return _firebase_app is not None
    _init_attempted = True

    if firebase_admin is None:
        _init_error = "firebase_admin_not_installed"
        logger.info("Firebase Admin SDK not installed; running without Firestore persistence.")
        return False

    settings = get_settings()
    if settings.demo_mode and not os.getenv("FORCE_FIREBASE"):
        _init_error = "demo_mode_disabled"
        logger.info("Firebase Admin disabled (DEMO_MODE=true). Set FORCE_FIREBASE=true to override.")
        return False

    options: Dict[str, Any] = {}
    if settings.firebase_project_id:
        options["projectId"] = settings.firebase_project_id

    try:
        if firebase_admin._apps:
            _firebase_app = firebase_admin.get_app()
        else:
            cred = _build_credential()
            _firebase_app = firebase_admin.initialize_app(credential=cred, options=options or None)

        if settings.firestore_database_id and settings.firestore_database_id != "(default)":
            _firestore_client = firebase_firestore.client(
                app=_firebase_app,
                database_id=settings.firestore_database_id,
            )
        else:
            _firestore_client = firebase_firestore.client(app=_firebase_app)

        _init_error = None
        logger.info(
            "Firebase Admin initialized. Firestore database: %s",
            settings.firestore_database_id or "(default)",
            extra={"event": "firebase_init_ok"},
        )
        return True
    except Exception as exc:
        _init_error = type(exc).__name__
        _firebase_app = None
        _firestore_client = None
        logger.warning(
            "Firebase init failed (%s): %s — falling back to in-memory store.",
            type(exc).__name__,
            str(exc),
            extra={"event": "firebase_init_fallback"},
        )
        return False


def get_firestore_client() -> Any:
    if not _init_attempted:
        init_firebase()
    return _firestore_client


def get_firebase_auth_module() -> Any:
    if not _init_attempted:
        init_firebase()
    if _firebase_app is None:
        return None
    return firebase_auth


def get_firebase_status() -> Dict[str, Any]:
    settings = get_settings()
    running_on_cloud_run = bool(os.getenv("K_SERVICE"))
    key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY", "").strip()
    auth_mode = (
        "service_account_key"
        if key_path and os.path.isfile(key_path)
        else "application_default_credentials"
    )
    return {
        "sdkInstalled": firebase_admin is not None,
        "initialized": _firebase_app is not None,
        "firestoreConnected": _firestore_client is not None,
        "projectIdConfigured": bool(settings.firebase_project_id),
        "firestoreDatabase": settings.firestore_database_id,
        "authMode": auth_mode,
        "cloudRunDetected": running_on_cloud_run,
        "status": "ready" if _firebase_app is not None else (_init_error or "uninitialized"),
    }


def reset_firebase_state_for_tests() -> None:
    global _firebase_app, _firestore_client, _init_attempted, _init_error
    _firebase_app = None
    _firestore_client = None
    _init_attempted = False
    _init_error = None
