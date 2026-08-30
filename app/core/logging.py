import json
import logging
import time
import uuid
from typing import Callable, Awaitable

from fastapi import Request

from app.core.config import get_settings

_settings = get_settings()

REQUEST_LOGGER = logging.getLogger("news_api.request")


def configure_logging() -> None:
    level_name = _settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    REQUEST_LOGGER.setLevel(level)


async def logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable],
):
    start = time.perf_counter()
    header_name = _settings.request_id_header
    request_id = request.headers.get(header_name) or uuid.uuid4().hex
    request.state.request_id = request_id
    extra = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
    }
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover - logged before re-raise
        duration_ms = (time.perf_counter() - start) * 1000.0
        log_payload = {
            **extra,
            "status": 500,
            "duration_ms": duration_ms,
            "error": repr(exc),
        }
        _log_json(log_payload, level="error")
        raise
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers[header_name] = request_id
    log_payload = {
        **extra,
        "status": response.status_code,
        "duration_ms": duration_ms,
    }
    _log_json(log_payload)
    return response


def _log_json(payload: dict, level: str = "info") -> None:
    if _settings.log_json:
        message = json.dumps(payload, separators=(",", ":"))
    else:
        message = "req_id=%s method=%s path=%s status=%s duration_ms=%.2f" % (
            payload.get("request_id"),
            payload.get("method"),
            payload.get("path"),
            payload.get("status"),
            payload.get("duration_ms", 0.0),
        )
    logger_method = getattr(REQUEST_LOGGER, level, REQUEST_LOGGER.info)
    logger_method(message)
