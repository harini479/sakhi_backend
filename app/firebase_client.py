"""Cloud Run–compatible Firebase Admin & Firestore initialization using ADC."""

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


def init_firebase() -> bool:
    """Initialize Firebase Admin SDK using Application Default Credentials (Cloud Run IAM).

    Never requires a local service-account JSON file in production.
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
        logger.info("Firebase Admin disabled for public buildathon demo mode.")
        return False
    options: Dict[str, Any] = {}
    if settings.firebase_project_id:
        options["projectId"] = settings.firebase_project_id

    try:
        if firebase_admin._apps:
            _firebase_app = firebase_admin.get_app()
        else:
            # Uses Application Default Credentials (Cloud Run runtime service account in GCP,
            # or `gcloud auth application-default login` locally). No JSON file required.
            cred = firebase_credentials.ApplicationDefault()
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
            "Firebase Admin initialized via Application Default Credentials.",
            extra={"event": "firebase_init_ok"},
        )
        return True
    except Exception as exc:
        _init_error = type(exc).__name__
        _firebase_app = None
        _firestore_client = None
        logger.warning(
            "Firebase Admin ADC initialization unavailable; falling back to local store.",
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
    return {
        "sdkInstalled": firebase_admin is not None,
        "initialized": _firebase_app is not None,
        "firestoreConnected": _firestore_client is not None,
        "projectIdConfigured": bool(settings.firebase_project_id),
        "authMode": "application_default_credentials",
        "cloudRunDetected": running_on_cloud_run,
        "status": "ready" if _firebase_app is not None else (_init_error or "uninitialized"),
    }


def reset_firebase_state_for_tests() -> None:
    global _firebase_app, _firestore_client, _init_attempted, _init_error
    _firebase_app = None
    _firestore_client = None
    _init_attempted = False
    _init_error = None
