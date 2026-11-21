"""Async LLM provider client supporting Gemini Developer API and optional OpenAI fallback."""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import get_settings
from .http_client import request_json

logger = logging.getLogger("sakhi.llm")

_last_provider_diagnostic: Optional[str] = None
_llm_call_timestamps: List[float] = []
MAX_LLM_CALLS_PER_MINUTE = 60


async def _call_openai(
    system_instruction: str,
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    global _last_provider_diagnostic
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    settings = get_settings()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]
    for turn in (history or [])[-6:]:
        content = str(turn.get("content", "")).strip()
        if content:
            mapped_role = "user" if turn.get("role") == "user" else "assistant"
            messages.append({"role": mapped_role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        status_code, data = await request_json(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json_body={
                "model": settings.openai_model,
                "temperature": 0.3,
                "max_tokens": 450,
                "messages": messages,
            },
            timeout=15.0,
        )
        if status_code != 200 or not isinstance(data, dict):
            _last_provider_diagnostic = f"openai_http_{status_code}"
            logger.warning(
                "OpenAI provider returned non-200 status.",
                extra={"event": "llm_openai_http_error", "status_code": status_code},
            )
            return None

        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "").strip()
            if content:
                _last_provider_diagnostic = None
                return content
    except Exception as exc:
        _last_provider_diagnostic = f"openai_{type(exc).__name__}"
        logger.warning(
            "OpenAI provider call failed.",
            extra={"event": "llm_openai_exception"},
        )
    return None


async def _call_gemini(
    system_instruction: str,
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    global _last_provider_diagnostic
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    settings = get_settings()
    history_lines: List[str] = []
    for turn in (history or [])[-4:]:
        content = str(turn.get("content", "")).strip()
        if content:
            speaker = "User" if turn.get("role") == "user" else "Sakhi"
            history_lines.append(f"{speaker}: {content}")

    prompt = (
        "Recent conversation:\n" + "\n".join(history_lines) + f"\n\nLatest user message:\n{user_message}"
        if history_lines
        else user_message
    )

    models_to_try = [
        (settings.gemini_model, "LOW", settings.gemini_timeout_seconds),
    ]
    if (
        settings.gemini_fallback_model
        and settings.gemini_fallback_model != settings.gemini_model
    ):
        models_to_try.append(
            (settings.gemini_fallback_model, "MINIMAL", settings.gemini_timeout_seconds + 2.0)
        )

    for model_name, thinking_level, timeout_sec in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": thinking_level},
                "maxOutputTokens": 600,
            },
        }
        try:
            status_code, data = await request_json(
                "POST",
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                    "User-Agent": "aistudio-build",
                },
                json_body=payload,
                timeout=timeout_sec,
            )
            if status_code != 200 or not isinstance(data, dict):
                _last_provider_diagnostic = f"gemini_http_{status_code}"
                logger.warning(
                    "Gemini API returned non-200 status; attempting fallback if available.",
                    extra={"event": "llm_gemini_http_error", "status_code": status_code},
                )
                continue

            candidates = data.get("candidates") or []
            if candidates:
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(
                    str(p.get("text", "")) for p in parts if not p.get("thought")
                ).strip()
                if text:
                    _last_provider_diagnostic = None
                    return text
        except Exception as exc:
            _last_provider_diagnostic = f"gemini_{type(exc).__name__}"
            logger.warning(
                "Gemini API call timed out or raised exception; attempting fallback if available.",
                extra={"event": "llm_gemini_exception"},
            )

    return None


async def generate_sakhi_reply(
    system_instruction: str,
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[str, str]]:
    """Generate a medically grounded reply with usage safeguards and call budgets."""
    global _llm_call_timestamps
    now = time.monotonic()
    _llm_call_timestamps = [t for t in _llm_call_timestamps if now - t < 60.0]
    if len(_llm_call_timestamps) >= MAX_LLM_CALLS_PER_MINUTE:
        logger.warning(
            "Global LLM rate limit budget exceeded; falling back to clinical knowledge.",
            extra={"event": "llm_budget_exceeded"},
        )
        return None

    _llm_call_timestamps.append(now)

    if os.getenv("OPENAI_API_KEY", "").strip():
        openai_text = await _call_openai(system_instruction, user_message, history)
        if openai_text:
            return openai_text, "openai"

    if os.getenv("GEMINI_API_KEY", "").strip():
        gemini_text = await _call_gemini(system_instruction, user_message, history)
        if gemini_text:
            return gemini_text, "gemini"

    return None


def get_active_llm_status() -> Dict[str, Any]:
    settings = get_settings()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )
    return {
        "openaiConfigured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "geminiConfigured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "geminiModel": settings.gemini_model,
        "supabaseConfigured": bool(supabase_url and supabase_key),
        "status": "degraded" if _last_provider_diagnostic else "ready",
    }
