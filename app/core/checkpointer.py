import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings

logger = logging.getLogger(__name__)

_checkpointer: AsyncPostgresSaver | None = None
_checkpointer_ctx = None


async def get_checkpointer() -> AsyncPostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Call init_checkpointer() first.")
    return _checkpointer


async def init_checkpointer() -> AsyncPostgresSaver:
    global _checkpointer, _checkpointer_ctx
    logger.info("Initializing AsyncPostgresSaver...")
    _checkpointer_ctx = AsyncPostgresSaver.from_conn_string(settings.database_url)
    _checkpointer = await _checkpointer_ctx.__aenter__()
    await _checkpointer.setup()
    logger.info("Checkpointer ready.")
    return _checkpointer


async def close_checkpointer() -> None:
    global _checkpointer, _checkpointer_ctx
    if _checkpointer_ctx:
        await _checkpointer_ctx.__aexit__(None, None, None)
        _checkpointer = None
        _checkpointer_ctx = None
        logger.info("Checkpointer closed.")
