import asyncio
import logging
import os

from langchain_core.messages import AIMessage
from langgraph.graph.message import MessagesState

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config")

_rails = None


def get_rails():
    global _rails
    if _rails is None:
        from nemoguardrails import LLMRails, RailsConfig
        config = RailsConfig.from_path(CONFIG_PATH)
        _rails = LLMRails(config)
    return _rails


async def guardrails_node(state: MessagesState) -> dict:
    messages = state["messages"]
    if not messages:
        return state

    last = messages[-1]
    if last.type != "human":
        return state

    user_text = last.content if isinstance(last.content, str) else ""

    try:
        rails = get_rails()
        result = await asyncio.wait_for(
            rails.generate_async(messages=[{"role": "user", "content": user_text}]),
            timeout=30.0,
        )
        if result and result != user_text:
            logger.info(f"Guardrails blocked: '{user_text[:60]}'")
            return {"messages": list(messages) + [AIMessage(content=result)]}
    except asyncio.TimeoutError:
        logger.warning("Guardrails timed out — allowing through")
    except Exception as e:
        logger.warning(f"Guardrails error (allowing through): {e}")

    return state
