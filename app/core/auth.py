import logging

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """
    Validates the API key from the request header.
    If no api_keys are configured (empty list), auth is disabled — local dev mode.
    In production, set API_KEYS=key1,key2,key3 in .env.
    """
    if not settings.api_keys:
        return "dev"

    if api_key is None or api_key not in settings.api_keys:
        logger.warning("Rejected request: invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
