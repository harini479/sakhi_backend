"""Optional live Supabase REST knowledge retriever with SSRF/HTTPS validation."""

import ipaddress
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .http_client import request_json


def _get_supabase_config() -> Optional[Tuple[str, str]]:
    raw_url = os.getenv("SUPABASE_URL", "").strip()
    api_key = (
        os.getenv("SUPABASE_SERVICE_ROLE", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )
    if not raw_url or not api_key:
        return None

    try:
        parsed = urlparse(raw_url)
        if parsed.scheme != "https" or not parsed.hostname:
            return None
        host = parsed.hostname.lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return None
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return None
        except ValueError:
            if host.endswith(".local") or host.endswith(".internal"):
                return None

        return raw_url.rstrip("/"), api_key
    except Exception:
        return None


def _sanitize_search_term(term: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s\u0900-\u097F\u0C00-\u0C7F]", " ", term or "").strip()


async def query_supabase_knowledge(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    cfg = _get_supabase_config()
    if not cfg:
        return []

    base_url, api_key = cfg
    cleaned = _sanitize_search_term(query)
    words = [w for w in cleaned.split() if len(w) >= 3][:4]
    if not words:
        return []

    primary_term = sorted(words, key=len, reverse=True)[0]
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    results: List[Dict[str, Any]] = []

    try:
        status_code, rows = await request_json(
            "GET",
            f"{base_url}/rest/v1/faq",
            headers=headers,
            params={
                "select": "question,answer,youtube_link,infographic_url",
                "or": f"(question.ilike.*{primary_term}*,answer.ilike.*{primary_term}*)",
                "limit": str(limit),
            },
            timeout=5.0,
        )
        if status_code == 200 and isinstance(rows, list):
            for row in rows:
                if row.get("question") and row.get("answer"):
                    results.append(
                        {
                            "source": "faq",
                            "title": row["question"],
                            "content": row["answer"],
                            "youtube_link": row.get("youtube_link"),
                            "infographic_url": row.get("infographic_url"),
                        }
                    )
    except Exception:
        pass

    if len(results) < limit:
        try:
            status_code, rows = await request_json(
                "GET",
                f"{base_url}/rest/v1/sakhi_knowledge_nodes",
                headers=headers,
                params={
                    "select": "title,content,display_text",
                    "or": f"(title.ilike.*{primary_term}*,content.ilike.*{primary_term}*)",
                    "limit": str(limit - len(results)),
                },
                timeout=5.0,
            )
            if status_code == 200 and isinstance(rows, list):
                for row in rows:
                    text = row.get("display_text") or row.get("content")
                    if row.get("title") and text:
                        results.append(
                            {
                                "source": "knowledge_nodes",
                                "title": row["title"],
                                "content": text,
                                "youtube_link": None,
                                "infographic_url": None,
                            }
                        )
        except Exception:
            pass

    return results
