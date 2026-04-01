import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_graph
from app.api.chat import router as chat_router
from app.api.resume import router as resume_router

configure_logging()
logger = logging.getLogger(__name__)

# Configure LangSmith tracing if enabled
if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

# Global references set during lifespan startup — accessed by chat.py
mcp_client: MultiServerMCPClient | None = None
agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_client, agent_graph

    logger.info("Starting up shelter-chat-api...")

    # 1. Init checkpointer — creates LangGraph tables in Postgres if needed
    checkpointer = await init_checkpointer()

    # 2. Connect to MCP server and load tools
    logger.info(f"Connecting to MCP server at {settings.mcp_server_url}...")
    mcp_client = MultiServerMCPClient({
        "shelter": {
            "url": settings.mcp_server_url,
            "transport": "streamable_http",
        }
    })
    tools = await asyncio.wait_for(mcp_client.get_tools(), timeout=10.0)
    logger.info(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    # 3. Build and compile the LangGraph agent
    agent_graph = build_graph(tools, checkpointer)
    logger.info("Agent graph ready. Startup complete.")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_checkpointer()
    if mcp_client:
        await mcp_client.aclose()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Shelter Chat API",
    description="Agentic social services chat powered by Claude + LangGraph + MCP",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(resume_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mcp_connected": mcp_client is not None,
        "agent_ready": agent_graph is not None,
    }
