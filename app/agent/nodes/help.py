import logging

from langchain_core.messages import AIMessage

from app.agent.state import NavigatorState

logger = logging.getLogger(__name__)

_HELP_TEXT = """\
Here's what I can help you with:

**Finding services** — Describe what your client needs and I'll search for matching shelters, food programs, health services, jobs, and more in San Francisco.

**Refining results** — After a search, ask me to narrow down by availability, location, specific eligibilities, or other criteria. For example: "Same but only open now" or "Actually she's a senior."

**Follow-up questions** — Ask about the results I found, compare options, or reason through next steps. For example: "Which of these is closest to the Tenderloin?" or "If the first one has a waitlist, what's the best alternative?"

**Organization lookups** — Ask about a specific org's hours, eligibility, contact info, or services. For example: "What are Glide's hours?" or "Does Compass Family accept pets?"

**Client context** — Tell me about your client and I'll use that to filter searches automatically. For example: "My client is a 45yo undocumented woman with 2 kids who speaks Spanish." I'll remember this across the conversation.

**To get started**, just describe what your client needs — for example:
"I need emergency shelter for a single adult male in the Tenderloin."\
"""


async def help_node(state: NavigatorState) -> dict:
    logger.info("help_node: returning capabilities response")
    return {"messages": [AIMessage(content=_HELP_TEXT)]}
