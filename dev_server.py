"""Runs the Sakhi Python backend via uvicorn (when installed) or Python 3.11 stdlib HTTP server."""

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import hospitals, store
from app.auth import AuthError, enforce_user_authorization, verify_webhook_secret
from app.chat_engine import process_sakhi_chat
from app.config import get_settings
from app.firebase_client import get_firebase_status, init_firebase
from app.llm import get_active_llm_status
from app.logging_config import configure_logging


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt.encode("utf-8"), n=16384, r=8, p=1)
    return f"scrypt${salt}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith("scrypt$"):
        parts = stored_hash.split("$")
        if len(parts) != 3:
            return False
        _, salt, expected_hex = parts
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt.encode("utf-8"), n=16384, r=8, p=1)
        return hmac.compare_digest(dk.hex(), expected_hex)
    return False


async def _handle_get(path: str, query: Dict[str, List[str]]) -> Tuple[int, Any]:
    if path == "/":
        return 200, {
            "status": "ok",
            "service": "backend-service",
            "health": "/health",
        }

    if path == "/health":
        return 200, {"status": "ok", "service": "backend-service"}

    if path == "/api/health":
        return 200, {
            "status": "ok",
            "service": "sakhi-whatsapp-backend",
            "runtime": "python",
            "message": "Sakhi WhatsApp & Clinical Companion Python Backend is running.",
            "providers": get_active_llm_status(),
            "firebase": get_firebase_status(),
        }

    if path == "/api/care-preview":
        return 200, store.get_predefined_care_preview()

    if path == "/api/hospitals":
        zipcode = (query.get("zipcode") or [""])[0].strip()
        found = await hospitals.find_nearby_hospitals(zipcode, "General", 10000)
        return 200, {"message": hospitals.build_hospital_list_message(found, zipcode)}

    if path == "/api/hospital_details":
        place_id = (query.get("place_id") or [""])[0].strip()
        details = await hospitals.get_hospital_details(place_id)
        return 200, {"message": hospitals.build_hospital_detail_card(details)}

    if path == "/api/proxy/knowledge_hub/recommendations":
        lens_lower = (query.get("lens") or ["medical"])[0].lower()
        lang_lower = (query.get("lang") or ["en"])[0].lower()
        try:
            limit = min(max(int((query.get("limit") or ["6"])[0]), 1), 20)
        except ValueError:
            limit = 6

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
        return 200, results

    return 404, {"detail": "Not found"}


async def _handle_post(
    path: str,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Any]:
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    authorization = hdrs.get("authorization")

    try:
        if path == "/api/chat":
            phone_number = store.normalize_phone(body.get("sessionPhone"))
            interactive_reply = body.get("interactiveReply") or {}
            message = str(body.get("message") or "")
            language = str(body.get("language") or "en")
            responses: List[Dict[str, Any]] = []

            def push_response(item: Any) -> None:
                if isinstance(item, str):
                    responses.append({"type": "text", "text": {"body": item}})
                elif item:
                    responses.append(item)

            if isinstance(interactive_reply, dict) and str(interactive_reply.get("id") or "").startswith("hospital_"):
                place_id = str(interactive_reply["id"]).replace("hospital_", "", 1)
                details = await hospitals.get_hospital_details(place_id)
                push_response(hospitals.build_hospital_detail_card(details))
                return 200, {"responses": responses}

            incoming_text = (
                str(interactive_reply.get("title"))
                if isinstance(interactive_reply, dict) and interactive_reply.get("title")
                else message
            )

            data = await process_sakhi_chat(
                user_id=None,
                phone_number=phone_number,
                message=incoming_text,
                language=language,
            )

            if data.get("error"):
                push_response(
                    "Please select a patient profile or start a guest session before sending a message."
                )
                return 200, {"responses": responses}

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

            return 200, {"responses": responses}

        if path == "/sakhi/chat":
            if body.get("user_id"):
                enforce_user_authorization(str(body.get("user_id")), authorization)
            result = await process_sakhi_chat(
                user_id=body.get("user_id"),
                phone_number=body.get("phone_number"),
                message=str(body.get("message") or ""),
                language=str(body.get("language") or "en"),
            )
            if result.get("statusCode") and result.get("error"):
                return int(result["statusCode"]), {"detail": result["error"]}
            return 200, result

        if path == "/api/workflow_step":
            return 200, {"handled": False}

        if path == "/user/register":
            name = str(body.get("name") or "").strip()
            email = body.get("email")
            phone_number = body.get("phone_number")
            password = body.get("password")
            if not name or (not email and not phone_number):
                return 400, {"detail": "name and email or phone_number are required"}

            identity = enforce_user_authorization(None, authorization) if authorization else None
            user_id = identity.uid if identity else str(uuid.uuid4())
            password_hash = _hash_password(str(password)) if password else None
            norm_phone = store.normalize_phone(phone_number) if phone_number else None
            norm_email = str(email).strip().lower() if email else None

            user_row = {
                "user_id": user_id,
                "id": user_id,
                "name": name,
                "email": norm_email,
                "phone_number": norm_phone,
                "role": "patient",
                "preferred_language": str(body.get("preferred_language") or "en"),
                "relation": str(body.get("user_relation") or "self"),
                "password_hash": password_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            store.save_user_profile(user_row)
            safe_user = {k: v for k, v in user_row.items() if k != "password_hash"}
            return 200, {"status": "success", "user_id": user_id, "user": safe_user}

        if path == "/user/login":
            email = str(body.get("email") or "").strip().lower()
            password = str(body.get("password") or "")
            if not email or not password:
                return 400, {"detail": "email and password are required"}
            user = store.users_by_email.get(email)
            if not user or not user.get("password_hash"):
                return 401, {"detail": "Invalid email or password"}
            if not _verify_password(password, str(user["password_hash"])):
                return 401, {"detail": "Invalid email or password"}
            safe_user = {k: v for k, v in user.items() if k != "password_hash"}
            return 200, {"status": "success", "user_id": user["user_id"], "user": safe_user}

        if path == "/user/answers":
            user_id = body.get("user_id")
            answers = body.get("answers")
            if not user_id or not isinstance(answers, list) or not answers:
                return 400, {"detail": "user_id and non-empty answers are required"}
            enforce_user_authorization(str(user_id), authorization)
            saved = store.save_user_answers(str(user_id), answers)
            return 200, {"status": "success", "saved": saved}

        if path == "/user/relation":
            user_id = body.get("user_id")
            relation = body.get("relation")
            if not user_id or not relation:
                return 400, {"detail": "user_id and relation are required"}
            enforce_user_authorization(str(user_id), authorization)
            store.update_user_profile(str(user_id), {"relation": str(relation)})
            return 200, {"status": "success"}

        if path == "/user/preferred-language":
            user_id = body.get("user_id")
            preferred_language = body.get("preferred_language")
            if not user_id or not preferred_language:
                return 400, {"detail": "user_id and preferred_language are required"}
            enforce_user_authorization(str(user_id), authorization)
            store.update_user_profile(str(user_id), {"preferred_language": str(preferred_language)})
            return 200, {"status": "success"}

        if path == "/webhook/whatsapp_incoming":
            verify_webhook_secret(hdrs.get("x-webhook-secret") or hdrs.get("x-internal-secret"))
            message_info = body.get("message_info")
            if not isinstance(message_info, dict):
                return 200, {"status": "error", "message": "No message_info provided"}
            message_id = message_info.get("message_id") or message_info.get("id") or body.get("id")
            if message_id and store.check_and_record_message_id(str(message_id)):
                return 200, {"status": "duplicate_ignored"}
            text = str(message_info.get("text") or "")
            phone = message_info.get("from")
            raw = message_info.get("raw") or {}
            list_reply = (raw.get("interactive") or {}).get("list_reply") or {}
            if list_reply.get("id") and str(list_reply["id"]).startswith("hospital_"):
                details = await hospitals.get_hospital_details(
                    str(list_reply["id"]).replace("hospital_", "", 1)
                )
                return 200, {"status": "success", "response": hospitals.build_hospital_detail_card(details)}
            if not text:
                return 200, {"status": "continue_chat"}
            response = await process_sakhi_chat(phone_number=phone, message=text, language="en")
            return 200, {"status": "success", "response": response}

        if path == "/expert/redirect":
            target_uid = str(body.get("user_id") or "").strip() or None
            enforce_user_authorization(target_uid, authorization)
            return 200, {
                "status": "pushed_to_dashboard",
                "message": "Your question has been sent to our medical experts.",
            }

        return 404, {"detail": "Not found"}
    except AuthError as auth_err:
        return auth_err.status_code, {"detail": auth_err.detail}


class SakhiRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status_code: int, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            status_code, payload = asyncio.run(_handle_get(parsed.path, query))
            self._send_json(status_code, payload)
        except Exception:
            self._send_json(500, {"detail": "Internal server error"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length > 2 * 1024 * 1024:
            self._send_json(413, {"detail": "Payload too large"})
            return
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            if not isinstance(body, dict):
                body = {}
        except Exception:
            self._send_json(400, {"detail": "Invalid JSON body"})
            return

        try:
            req_headers = {k: v for k, v in self.headers.items()}
            status_code, payload = asyncio.run(_handle_post(parsed.path, body, req_headers))
            self._send_json(status_code, payload)
        except Exception:
            self._send_json(500, {"detail": "Internal server error"})


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_firebase()

    host = os.getenv("PYTHON_BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("PYTHON_BACKEND_PORT") or os.getenv("PORT") or "8001")

    try:
        import uvicorn  # type: ignore
        import main  # type: ignore

        uvicorn.run(main.app, host=host, port=port, log_level="warning")
        return
    except ImportError:
        pass

    server = ThreadingHTTPServer((host, port), SakhiRequestHandler)
    print(f"[Python Backend] Starting Python 3.11 server on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
