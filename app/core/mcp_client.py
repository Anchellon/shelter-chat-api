import asyncio
import json
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def _unwrap_result(result) -> list | dict:
    """
    Normalize MCP content blocks to plain Python objects.

    The MCP protocol returns tool results as content blocks:
        [{"type": "text", "text": "<json string>", "id": "..."}]

    langchain_mcp_adapters passes these through as-is. This unwraps them
    so all application code receives clean deserialized objects.
    """
    if isinstance(result, list) and result and isinstance(result[0], dict):
        first = result[0]
        if first.get("type") == "text" and isinstance(first.get("text"), str):
            try:
                return json.loads(first["text"])
            except json.JSONDecodeError:
                logger.warning("MCP tool result: type=text but content is not valid JSON")
    return result


class MCPClient:
    """
    Wrapper around MultiServerMCPClient that manages connection lifecycle
    and normalizes all tool results to clean Python objects.
    """

    def __init__(self, server_url: str):
        self._url = server_url
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, object] = {}

    async def connect(self, timeout: float = 10.0) -> None:
        self._client = MultiServerMCPClient({
            "shelter": {
                "url": self._url,
                "transport": "streamable_http",
            }
        })
        tools = await asyncio.wait_for(self._client.get_tools(), timeout=timeout)
        self._tools = {t.name: t for t in tools}
        logger.info(f"MCPClient connected — {len(tools)} tools: {list(self._tools)}")

    async def close(self) -> None:
        self._client = None

    @property
    def tools(self) -> list:
        """Raw LangChain tool objects — passed to the agent graph."""
        return list(self._tools.values())

    async def invoke(self, tool_name: str, args: dict):
        """Invoke a tool by name and return a clean Python object."""
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"MCP tool '{tool_name}' not available")
        result = await tool.ainvoke(args)
        return _unwrap_result(result)
