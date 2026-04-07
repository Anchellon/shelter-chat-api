import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/services", tags=["services"])


class ServicesBatchRequest(BaseModel):
    service_ids: list[int]


@router.post("/batch")
async def services_batch(
    request: ServicesBatchRequest,
    _: str = Depends(require_user),
):
    from app.main import mcp_client

    if not request.service_ids:
        raise HTTPException(status_code=400, detail="service_ids cannot be empty")

    if mcp_client is None:
        raise HTTPException(status_code=503, detail="MCP client not available")

    logger.info(f"POST /services/batch — {len(request.service_ids)} ids")

    try:
        services = await mcp_client.invoke(
            "get_service_details_batch",
            {"service_ids": request.service_ids},
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"services": services}
