import logging

from langchain_core.messages import AIMessage

from app.agent.state import NavigatorState

logger = logging.getLogger(__name__)

_HELP_TEXT = """\
Here's what I can help you with:

**Service categories I can search** (in San Francisco):
- Shelter and emergency housing
- Long-term housing and rental support
- Food (meals, food pantries, groceries)
- Health and medical care
- Mental health and counseling
- Substance use treatment and harm reduction
- Jobs and employment support
- Legal aid and immigration services
- Domestic violence and crisis support
- Clothing, hygiene, and other basic needs

**Finding services** — Describe what your client needs and I'll search for matching options. Example: "I need emergency shelter for a single adult male in the Tenderloin."

**Refining results** — After a search, ask me to narrow down by availability, location, eligibility, or other criteria. Example: "Same but only open now" or "Actually she's a senior."

**Follow-up questions** — Ask about the results I found, compare options, or reason through next steps. Example: "Which of these is closest to the Tenderloin?" or "If the first one has a waitlist, what's the best alternative?"

**Organization lookups** — Ask about a specific org's hours, eligibility, contact info, or services. Example: "What are Glide's hours?" or "Does Compass Family accept pets?"

**Client context** — Tell me about your client and I'll use that to filter searches automatically. Example: "My client is a 45yo undocumented woman with 2 kids who speaks Spanish." I'll remember this across the conversation.

**To get started**, just describe what your client needs — or pick a category above and I'll search for it.\
"""


async def help_node(state: NavigatorState) -> dict:
    logger.info("help_node: returning capabilities response")
    return {"messages": [AIMessage(content=_HELP_TEXT)]}
