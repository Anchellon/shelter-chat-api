import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.state import NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are assisting a social services navigator. Their message is ambiguous — it could mean:
1. A structured search for services for a specific client (returns a list of matching services with referral options)
2. General information about what services or organizations exist (returns a text answer)

Ask ONE short, clear follow-up question to determine which they want.

Keep it to one sentence. Do not ask multiple questions. Do not explain your reasoning.

Examples:
- "Are you looking to find services for a specific client right now, or do you want general information about what's available?"
- "Would you like me to search for options you can refer a client to, or are you looking for general info about these services?"\
"""


async def clarify_node(state: NavigatorState) -> dict:
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )

    llm = get_llm(settings.formatter_provider, settings.formatter_model)

    context = f'Navigator said: "{last_human.content}"' if last_human else "Navigator sent an ambiguous message."

    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    logger.info("clarify_node: asking clarifying question")
    return {"messages": [AIMessage(content=response.content)], "pending_action": "clarify"}
