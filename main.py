"""Sakhi Fertility & Post-Care Clinical Companion — Cloud Run FastAPI Backend."""

import logging
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import hospitals, store
from app.auth import AuthError, enforce_user_authorization, verify_webhook_secret
from app.chat_engine import process_sakhi_chat
from app.config import get_settings
from app.firebase_client import get_firebase_status, init_firebase
from app.llm import get_active_llm_status
from app.logging_config import configure_logging

load_dotenv()
settings = get_settings()
logger = configure_logging(settings.log_level)

# In-memory sliding-window IP rate limiter
_ip_rate_tracker: Dict[str, List[float]] = {}


def check_ip_rate_limit(client_ip: str, max_per_minute: int = 30) -> bool:
    now = time.monotonic()
    timestamps = _ip_rate_tracker.setdefault(client_ip, [])
    # Prune timestamps older than 60 seconds
    valid = [t for t in timestamps if now - t < 60.0]
    _ip_rate_tracker[client_ip] = valid
    if len(valid) >= max_per_minute:
        return False
    valid.append(now)
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    current = get_settings()
    configure_logging(current.log_level)
    init_firebase()
    logger.info(
        "Sakhi FastAPI backend started.",
        extra={"event": "app_startup"},
    )
    yield
    logger.info(
        "Sakhi FastAPI backend shutting down.",
        extra={"event": "app_shutdown"},
    )


app = FastAPI(
    title="Sakhi Clinical Companion API",
    description="Multilingual Pre-Care & Post-Care fertility, pregnancy, and clinical companion backend.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# Strictly scoped CORS from CORS_ORIGINS / ALLOWED_ORIGINS (no wildcard in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=bool(settings.cors_origins and "*" not in settings.cors_origins),
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-Id", "X-Session-Id"],
)


# ---------------------------------------------------------------------------
# Structured Non-PHI Request Logging & Safe Global Exception Handling
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request.state.request_id = request_id

    # Enforce request body size limit
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_request_body_bytes:
                logger.warning(
                    "Request rejected: payload size exceeds limit.",
                    extra={"event": "payload_too_large", "bytes": content_length},
                )
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Payload too large. Maximum allowed size is 32 KB.", "request_id": request_id},
                    headers={"X-Request-Id": request_id},
                )
        except ValueError:
            pass

    started = time.perf_counter()
    try:
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "Request completed.",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response
    except Exception:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "Unhandled server exception.",
            extra={
                "event": "http_unhandled_error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": latency_ms,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-Id": request_id},
        )


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
        headers={"X-Request-Id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={"X-Request-Id": request_id},
    )


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------
class InteractiveReply(BaseModel):
    id: str = Field(..., max_length=200)
    title: Optional[str] = Field(default=None, max_length=300)


class UnifiedChatRequest(BaseModel):
    message: str = Field(default="", max_length=1000)
    sessionPhone: Optional[str] = Field(default=None, max_length=64)
    sessionId: Optional[str] = Field(default=None, max_length=128)
    interactiveReply: Optional[InteractiveReply] = None
    language: Optional[str] = Field(default="en", max_length=32)


class DirectChatRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, max_length=128)
    phone_number: Optional[str] = Field(default=None, max_length=64)
    message: str = Field(default="", max_length=1000)
    language: Optional[str] = Field(default="en", max_length=32)


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: Optional[str] = Field(default=None, max_length=200)
    phone_number: Optional[str] = Field(default=None, max_length=64)
    password: Optional[str] = Field(default=None, min_length=6, max_length=128)
    role: str = Field(default="patient", max_length=40)
    preferred_language: str = Field(default="en", max_length=32)
    user_relation: str = Field(default="self", max_length=40)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=200)
    password: str = Field(..., min_length=1, max_length=128)


class UserAnswersRequest(BaseModel):
    user_id: str = Field(..., max_length=128)
    answers: List[Dict[str, Any]] = Field(..., min_length=1)


class UserRelationRequest(BaseModel):
    user_id: str = Field(..., max_length=128)
    relation: str = Field(..., max_length=64)


class UserLanguageRequest(BaseModel):
    user_id: str = Field(..., max_length=128)
    preferred_language: str = Field(..., max_length=32)


# ---------------------------------------------------------------------------
# Health & Predefined Care Data Preview (Zero Live LLM Calls)
# ---------------------------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "backend-service",
        "health": "/health",
    }


@app.get("/health")
async def cloud_run_health() -> Dict[str, str]:
    """Lightweight Cloud Run startup/liveness/readiness health endpoint."""
    return {"status": "ok", "service": "backend-service"}


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "sakhi-whatsapp-backend",
        "runtime": "python-fastapi",
        "message": "Sakhi WhatsApp & Clinical Companion FastAPI Backend is running.",
        "providers": get_active_llm_status(),
        "firebase": get_firebase_status(),
    }


@app.get("/api/care-preview")
async def get_care_preview() -> Dict[str, Any]:
    return store.get_predefined_care_preview()


# ---------------------------------------------------------------------------
# Unified Chat Endpoint (/api/chat) & Direct Endpoint (/sakhi/chat)
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def unified_chat(
    payload: UnifiedChatRequest,
    request: Request,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "127.0.0.1"

    if not check_ip_rate_limit(client_ip, settings.rate_limit_per_minute):
        logger.warning(
            "Rate limit exceeded on chat endpoint.",
            extra={"event": "rate_limit_exceeded", "ip": client_ip},
        )
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please wait a moment before sending another message.", "request_id": request_id},
            headers={"Retry-After": "60", "X-Request-Id": request_id},
        )

    # Validate or generate unpredictable guest session identifier (128-bit entropy)
    session_id = payload.sessionId or request.headers.get("X-Session-Id")
    if not session_id or not re.match(r"^guest_[a-zA-Z0-9_-]{12,64}$", session_id):
        if payload.sessionPhone and payload.sessionPhone.startswith("guest_"):
            session_id = payload.sessionPhone
        else:
            session_id = f"guest_{secrets.token_hex(16)}"

    phone_number = store.normalize_phone(payload.sessionPhone)
    # If phone is not a recognized synthetic patient template, isolate as pure guest
    if phone_number and not phone_number.startswith("guest_") and not store.get_patient_by_phone(phone_number):
        phone_number = None

    responses: List[Dict[str, Any]] = []

    def push_response(item: Any) -> None:
        if isinstance(item, str):
            responses.append({"type": "text", "text": {"body": item}, "sessionId": session_id})
        elif item:
            if isinstance(item, dict):
                item.setdefault("sessionId", session_id)
            responses.append(item)

    if payload.interactiveReply and payload.interactiveReply.id.startswith("hospital_"):
        place_id = payload.interactiveReply.id.replace("hospital_", "", 1)
        details = await hospitals.get_hospital_details(place_id)
        push_response(hospitals.build_hospital_detail_card(details))
        return JSONResponse(
            content={"responses": responses, "sessionId": session_id},
            headers={"X-Session-Id": session_id, "X-Request-Id": request_id},
        )

    incoming_text = (
        payload.interactiveReply.title
        if payload.interactiveReply and payload.interactiveReply.title
        else payload.message or ""
    )

    data = await process_sakhi_chat(
        user_id=None,
        phone_number=phone_number,
        message=incoming_text,
        language=payload.language or "en",
        session_id=session_id,
    )

    if data.get("error"):
        push_response(
            "Please select a patient profile or start a guest session before sending a message."
        )
        return JSONResponse(
            content={"responses": responses, "sessionId": session_id},
            headers={"X-Session-Id": session_id, "X-Request-Id": request_id},
        )

    is_hospital_search = (
        data.get("route") == "hospital_search" and data.get("intent") == "hospital search"
    )

    if data.get("reply") and not is_hospital_search:
        push_response(
            {
                "type": "text",
                "text": {"body": data["reply"]},
                "metadata": {
                    "mode": data.get("mode"),
                    "intent": data.get("intent"),
                    "route": data.get("route"),
                    "language": data.get("language"),
                    "sessionId": session_id,
                },
            }
        )

    if data.get("youtube_link"):
        push_response(
            {"type": "text", "text": {"body": data["youtube_link"], "preview_url": True}}
        )

    if data.get("infographic_url"):
        push_response({"type": "image", "image": {"link": data["infographic_url"]}})

    if is_hospital_search:
        zip_match = re.search(r"\b\d{6}\b", incoming_text)
        zipcode = data.get("zip_code") or (zip_match.group(0) if zip_match else None)
        if zipcode:
            nearby = await hospitals.find_nearby_hospitals(zipcode, "General", 10000)
            push_response(hospitals.build_hospital_list_message(nearby, zipcode))

    return JSONResponse(
        content={"responses": responses, "sessionId": session_id},
        headers={"X-Session-Id": session_id, "X-Request-Id": request_id},
    )


@app.post("/sakhi/chat")
async def direct_sakhi_chat(
    payload: DirectChatRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "127.0.0.1"

    if not check_ip_rate_limit(client_ip, settings.rate_limit_per_minute):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please wait a moment before sending another message.", "request_id": request_id},
            headers={"Retry-After": "60", "X-Request-Id": request_id},
        )

    session_id = request.headers.get("X-Session-Id") or f"guest_{secrets.token_hex(16)}"
    if payload.user_id:
        enforce_user_authorization(payload.user_id, authorization)

    result = await process_sakhi_chat(
        user_id=payload.user_id,
        phone_number=payload.phone_number,
        message=payload.message,
        language=payload.language or "en",
        session_id=session_id,
    )
    if result.get("statusCode") and result.get("error"):
        raise HTTPException(status_code=int(result["statusCode"]), detail=result["error"])
    return JSONResponse(
        content=result,
        headers={"X-Session-Id": session_id, "X-Request-Id": request_id},
    )


@app.post("/api/workflow_step")
async def workflow_step() -> Dict[str, bool]:
    raise HTTPException(status_code=404, detail="Endpoint not available in demo mode")


# ---------------------------------------------------------------------------
# Hospital Discovery & Knowledge Hub Recommendations
# ---------------------------------------------------------------------------
@app.get("/api/hospitals")
async def list_hospitals(zipcode: str = Query(default="", max_length=20)) -> Dict[str, Any]:
    cleaned = zipcode.strip()
    found = await hospitals.find_nearby_hospitals(cleaned, "General", 10000)
    return {"message": hospitals.build_hospital_list_message(found, cleaned)}


@app.get("/api/hospital_details")
async def hospital_details(place_id: str = Query(default="", max_length=200)) -> Dict[str, Any]:
    details = await hospitals.get_hospital_details(place_id.strip())
    return {"message": hospitals.build_hospital_detail_card(details)}


@app.get("/api/proxy/knowledge_hub/recommendations")
async def knowledge_hub_recommendations(
    lens: str = Query(default="medical", max_length=40),
    lang: str = Query(default="en", max_length=20),
    limit: int = Query(default=6, ge=1, le=20),
) -> List[Dict[str, Any]]:
    lens_lower = lens.lower()
    lang_lower = lang.lower()
    results = []
    for item in store.medical_knowledge:
        if lens_lower == "all":
            matched = True
        elif lens_lower == "social":
            matched = item["lens"] == "emotional"
        else:
            matched = item["lens"] == lens_lower or lens_lower in item["category"].lower()
        if matched:
            results.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "summary": item["content"],
                    "category": item["lens"] or item["category"],
                    "sub_category": item["sub_category"],
                    "language": lang_lower,
                    "infographic_url": None,
                    "youtube_link": None,
                }
            )
        if len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------------------
# User Registration, Login & Protected User-Scoped Preferences
# ---------------------------------------------------------------------------
@app.post("/user/register")
async def register_user(
    payload: RegisterRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if settings.demo_mode:
        raise HTTPException(status_code=403, detail="User registration is disabled in public demo mode.")

    if not payload.email and not payload.phone_number:
        raise HTTPException(status_code=400, detail="email or phone_number is required")

    identity = enforce_user_authorization(None, authorization) if authorization else None
    user_id = identity.uid if identity else str(uuid.uuid4())

    password_hash: Optional[str] = None
    if payload.password:
        hashed = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt(rounds=10))
        password_hash = hashed.decode("utf-8")

    norm_phone = store.normalize_phone(payload.phone_number) if payload.phone_number else None
    norm_email = payload.email.strip().lower() if payload.email else None

    # Prevent client-supplied privilege escalation; always register standard patient role
    user_row = {
        "user_id": user_id,
        "id": user_id,
        "name": payload.name.strip(),
        "email": norm_email,
        "phone_number": norm_phone,
        "role": "patient",
        "preferred_language": payload.preferred_language,
        "relation": payload.user_relation,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    store.save_user_profile(user_row)
    safe_user = {k: v for k, v in user_row.items() if k != "password_hash"}
    return {"status": "success", "user_id": user_id, "user": safe_user}


@app.post("/user/login")
async def login_user(payload: LoginRequest) -> Dict[str, Any]:
    if settings.demo_mode:
        raise HTTPException(status_code=403, detail="User login is disabled in public demo mode.")

    user = store.users_by_email.get(payload.email.strip().lower())
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not bcrypt.checkpw(
        payload.password.encode("utf-8"),
        str(user["password_hash"]).encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return {"status": "success", "user_id": user["user_id"], "user": safe_user}


@app.post("/user/answers")
async def save_user_answers(
    payload: UserAnswersRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if settings.demo_mode:
        raise HTTPException(status_code=403, detail="Patient record persistence is disabled in public demo mode.")
    enforce_user_authorization(payload.user_id, authorization)
    saved = store.save_user_answers(payload.user_id, payload.answers)
    return {"status": "success", "saved": saved}


@app.post("/user/relation")
async def update_user_relation(
    payload: UserRelationRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    if settings.demo_mode:
        raise HTTPException(status_code=403, detail="Patient record updates are disabled in public demo mode.")
    enforce_user_authorization(payload.user_id, authorization)
    store.update_user_profile(payload.user_id, {"relation": payload.relation})
    return {"status": "success"}


@app.post("/user/preferred-language")
async def update_user_language(
    payload: UserLanguageRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    if settings.demo_mode:
        raise HTTPException(status_code=403, detail="Patient record updates are disabled in public demo mode.")
    enforce_user_authorization(payload.user_id, authorization)
    store.update_user_profile(payload.user_id, {"preferred_language": payload.preferred_language})
    return {"status": "success"}


# ---------------------------------------------------------------------------
# WhatsApp Webhook & Expert Escalation
# ---------------------------------------------------------------------------
@app.post("/webhook/whatsapp_incoming")
async def whatsapp_incoming(
    body: Dict[str, Any],
    x_webhook_secret: Optional[str] = Header(default=None),
    x_internal_secret: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    verify_webhook_secret(x_webhook_secret or x_internal_secret)

    message_info = body.get("message_info")
    if not isinstance(message_info, dict):
        return {"status": "error", "message": "No message_info provided"}

    message_id = message_info.get("message_id") or message_info.get("id") or body.get("id")
    if message_id and store.check_and_record_message_id(str(message_id)):
        return {"status": "duplicate_ignored"}

    text = str(message_info.get("text") or "")
    phone = message_info.get("from")
    raw = message_info.get("raw") or {}
    list_reply = (raw.get("interactive") or {}).get("list_reply") or {}

    if list_reply.get("id") and str(list_reply["id"]).startswith("hospital_"):
        details = await hospitals.get_hospital_details(
            str(list_reply["id"]).replace("hospital_", "", 1)
        )
        return {"status": "success", "response": hospitals.build_hospital_detail_card(details)}

    if not text:
        return {"status": "continue_chat"}

    response = await process_sakhi_chat(phone_number=phone, message=text, language="en")
    return {"status": "success", "response": response}


@app.post("/expert/redirect")
async def expert_redirect(
    body: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    if settings.demo_mode:
        raise HTTPException(status_code=403, detail="Expert dashboard redirection is disabled in public demo mode.")
    target_uid = str(body.get("user_id") or "").strip() or None
    enforce_user_authorization(target_uid, authorization)
    return {
        "status": "pushed_to_dashboard",
        "message": "Your question has been sent to our medical experts.",
    }
