"""Async HTTP JSON client using httpx when available, with Python stdlib urllib fallback."""

import asyncio
import json
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import httpx  # type: ignore
except ImportError:
    httpx = None  # type: ignore


def _urllib_request_sync(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Tuple[int, Any]:
    full_url = f"{url}?{urlencode(params)}" if params else url
    req_headers = dict(headers or {})
    data: Optional[bytes] = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = Request(full_url, data=data, headers=req_headers, method=method.upper())
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace") if err.fp else ""
        try:
            return err.code, json.loads(raw)
        except Exception:
            return err.code, raw


async def request_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Tuple[int, Any]:
    if httpx is not None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method.upper(),
                url,
                headers=headers,
                params=params,
                json=json_body,
            )
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text

    return await asyncio.to_thread(
        _urllib_request_sync,
        method,
        url,
        headers,
        params,
        json_body,
        timeout,
    )
