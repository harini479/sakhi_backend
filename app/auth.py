"""Firebase ID token verification and cross-user authorization guards."""

import hmac
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .config import get_settings
from .firebase_client import get_firebase_auth_module

logger = logging.getLogger("sakhi.auth")


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class AuthenticatedIdentity:
    uid: str
    email: Optional[str] = None
    phone_number: Optional[str] = None
    role: str = "patient"


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1]:
        return parts[1]
    return None


def verify_firebase_id_token(
    authorization: Optional[str],
    *,
    required: bool = False,
    token_verifier: Optional[Callable[[str], dict]] = None,
) -> Optional[AuthenticatedIdentity]:
    """Verify a Firebase ID token from the Authorization header.

    Never assumes an email address alone proves identity; identity is derived
    exclusively from the cryptographically verified Firebase JWT `uid` claim.
    """
    token = extract_bearer_token(authorization)
    if not token:
        if required:
            raise AuthError(401, "Missing or malformed Authorization Bearer token")
        return None

    verifier = token_verifier
    if verifier is None:
        fb_auth = get_firebase_auth_module()
        if fb_auth is None:
            raise AuthError(503, "Authentication provider is not configured on this instance")
        verifier = lambda raw_tok: fb_auth.verify_id_token(raw_tok, check_revoked=False)

    try:
        decoded = verifier(token)
    except Exception:
        logger.warning("Rejected invalid or expired Firebase ID token.", extra={"event": "auth_invalid_token"})
        raise AuthError(401, "Invalid or expired authentication token")

    uid = str(decoded.get("uid") or decoded.get("sub") or "").strip()
    if not uid:
        raise AuthError(401, "Authentication token missing subject identity")

    return AuthenticatedIdentity(
        uid=uid,
        email=decoded.get("email"),
        phone_number=decoded.get("phone_number"),
        role=str(decoded.get("role") or "patient"),
    )


def enforce_user_authorization(
    target_user_id: Optional[str],
    authorization: Optional[str],
    *,
    token_verifier: Optional[Callable[[str], dict]] = None,
) -> Optional[AuthenticatedIdentity]:
    """Enforce Firebase token auth and prevent cross-user data access.

    - In production (or when REQUIRE_FIREBASE_AUTH=true), a valid Bearer token is required.
    - Whenever a Bearer token is present (in any environment), it is cryptographically verified
      and `identity.uid` MUST match `target_user_id` unless the caller holds an admin claim.
    """
    settings = get_settings()
    must_authenticate = settings.require_firebase_auth or bool(authorization)
    identity = verify_firebase_id_token(
        authorization,
        required=must_authenticate,
        token_verifier=token_verifier,
    )

    if identity and target_user_id:
        is_admin = identity.role.lower() in {"admin", "clinician"}
        if not is_admin and identity.uid != target_user_id:
            logger.warning(
                "Blocked cross-user access attempt.",
                extra={"event": "auth_cross_user_denied"},
            )
            raise AuthError(403, "Forbidden: cannot access or modify another user's records")

    return identity


def verify_webhook_secret(provided_secret: Optional[str]) -> None:
    """Verify shared webhook secret. In public demo mode, webhook requires configured secret."""
    settings = get_settings()
    expected = settings.webhook_secret
    if not expected:
        if settings.demo_mode or settings.is_production:
            raise AuthError(403, "Webhook endpoint is disabled in public demo mode")
        return
    if not provided_secret or not hmac.compare_digest(provided_secret.strip(), expected):
        logger.warning("Rejected webhook call with invalid secret.", extra={"event": "webhook_unauthorized"})
        raise AuthError(401, "Unauthorized webhook request")
