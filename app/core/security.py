from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_settings = get_settings()
_api_key_scheme = APIKeyHeader(name=_settings.api_key_header, auto_error=False)


def require_api_key(api_key: str | None = Depends(_api_key_scheme)) -> None:
    settings = get_settings()
    allowed_keys = settings.allowed_api_keys()
    if not allowed_keys:
        # No keys configured; treat endpoint as open.
        return
    if api_key in allowed_keys:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
