import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.state import ClientContext, NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are updating a client profile for a social services navigator.

The navigator's message may:
1. SET new client attributes — store them in the appropriate fields
2. UPDATE existing attributes — modify specific fields only
3. CLEAR the entire context — when they say "new client", "clear context", "reset", "different client", "next client"

Map information to these fields (omit a field entirely if it is not mentioned and should not change):
- age: e.g. "45yo adult", "senior", "teenager", "child under 10"
- housing: e.g. "unhoused", "near homeless", "at risk of eviction", "in shelter"
- gender: e.g. "woman", "man", "LGBTQ+", "non-binary", "transgender woman"
- family_status: e.g. "single parent with 2 kids", "pregnant", "married no children", "individuals"
- employment: e.g. "unemployed", "veteran", "retired", "employed part-time"
- financial: e.g. "low-income", "uninsured", "on SSI"
- health: e.g. "substance dependency", "HIV positive", "chronic illness", "mental health concerns"
- ethnicity: e.g. "Latinx", "Filipino/a", "Chinese", "African/Black"
- immigration: e.g. "undocumented", "asylum seeker", "DACA", "recent immigrant"
- language: e.g. "Spanish only", "Cantonese", "Mandarin", "limited English"
- other: e.g. "DV survivor", "SF resident", "trauma survivor", "human trafficking survivor"

Also detect if the message implies an obvious next action:
- If the client clearly needs a specific service (housing, food, shelter, health, jobs, etc.) → set pending_action to "new_search"
- Otherwise → set pending_action to null

Return ONLY a JSON object. No explanation. No markdown fences.

Format:
{
  "action": "update" | "clear",
  "fields": { ...only fields that change... },
  "pending_action": "new_search" | null,
  "confirmation": "brief 1-2 sentence confirmation to navigator. If pending_action is new_search, end with a confirmation question like 'Want me to search for options?'"
}

Examples:

Message: "My client is a 45yo undocumented woman with 2 kids who speaks Spanish"
Output: {"action": "update", "fields": {"age": "45yo adult", "gender": "woman", "immigration": "undocumented", "family_status": "2 kids", "language": "Spanish only"}, "pending_action": null, "confirmation": "Got it — 45yo undocumented woman with 2 kids, Spanish-speaking."}

Message: "She needs emergency housing"
Output: {"action": "update", "fields": {"housing": "needs emergency housing"}, "pending_action": "new_search", "confirmation": "Got it — added housing need. Want me to search for emergency housing options?"}

Message: "Actually she's a senior, not 45"
Output: {"action": "update", "fields": {"age": "senior"}, "pending_action": null, "confirmation": "Updated — client is a senior."}

Message: "New client"
Output: {"action": "clear", "fields": {}, "pending_action": null, "confirmation": "Client context cleared. Ready for a new client."}\
"""


def _merge_context(existing: ClientContext | None, fields: dict) -> ClientContext:
    base: ClientContext = dict(existing) if existing else {}
    for key, value in fields.items():
        if value is None:
            base.pop(key, None)
        else:
            base[key] = value
    return base  # type: ignore[return-value]


async def update_client_context_node(state: NavigatorState) -> dict:
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        logger.warning("update_client_context: no human message in state")
        return {}

    current_context = state.get("client_context")
    current_context_str = json.dumps(current_context) if current_context else "None"

    llm = get_llm(settings.classifier_provider, settings.classifier_model, json_mode=True)

    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Current context: {current_context_str}\n\nNavigator: {last_human.content}"),
    ])

    raw = response.content if isinstance(response.content, str) else "{}"
    logger.info(f"update_client_context raw: {raw[:300]}")

    try:
        text = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
    except Exception as e:
        logger.error(f"update_client_context parse error: {e} | raw: {raw[:300]}")
        parsed = {}

    action = parsed.get("action", "update")
    fields = parsed.get("fields", {})
    pending_action = parsed.get("pending_action")
    confirmation = parsed.get("confirmation", "Got it.")

    if action == "clear":
        logger.info("update_client_context: clearing context + zeroing search state")
        return {
            "client_context": None,
            "pending_action": pending_action,
            "groups": [],
            "results": {},
            "formatted": {},
            "messages": [AIMessage(content=confirmation)],
        }

    new_context = _merge_context(current_context, fields)
    logger.info(f"update_client_context: updated context → {new_context}")

    return {
        "client_context": new_context,
        "pending_action": pending_action,
        "messages": [AIMessage(content=confirmation)],
    }
