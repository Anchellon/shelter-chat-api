import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.state import NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

_VALID_INTENTS = frozenset({
    "new_search", "refine", "follow_up", "query",
    "set_context", "help", "acknowledge", "clarify",
})

_SYSTEM_PROMPT = """\
You are an intent classifier for a social services navigator assistant.

Classify the navigator's message into exactly ONE primary intent:

- new_search: They want to find services for a client (fresh query describing needs)
- refine: They want to modify the prior search ("same but open now", "actually she's a senior", "add food")
- follow_up: They have a question about results already found ("which is closest?", "summarize", "compare these two")
- query: They're asking about a specific named org ("what are Glide's hours?", "does Compass accept pets?")
- set_context: They're providing or updating client demographics ("my client is a 45yo woman", "new client", "she's also pregnant")
- help: They want to know what the assistant can do ("what can you do?", "help", "how does this work?")
- acknowledge: Confirming or reacting without requesting action ("ok", "thanks", "got it", "sounds good", "yes" with no context)
- clarify: Message is genuinely too ambiguous to classify — use sparingly, only when truly impossible

Rules:
- Prefer a concrete intent over "clarify" — only resort to clarify if a reasonable guess is impossible
- "refine" only applies when there are prior search groups to modify
- "follow_up" only applies when there are prior search results to reference
- If the message combines set_context with a service need ("my client is 45 and needs food"), primary = "set_context", secondary_intent = "new_search", secondary_message = the service need portion
- Only set secondary_intent when the message clearly contains two distinct actionable intents
{pending_action_context}
Prior search state: {prior_state}

Return ONLY a JSON object. No explanation. No markdown fences.
Format: {{"intent": "...", "secondary_intent": "..." | null, "secondary_message": "..." | null}}

Examples:
Message: "I need emergency shelter for a single adult male"
Output: {{"intent": "new_search", "secondary_intent": null, "secondary_message": null}}

Message: "Same but only open now"
Output: {{"intent": "refine", "secondary_intent": null, "secondary_message": null}}

Message: "Which of those shelters is closest to the Tenderloin?"
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "What are Glide's current hours?"
Output: {{"intent": "query", "secondary_intent": null, "secondary_message": null}}

Message: "My client is a 45yo undocumented woman with 2 kids who needs food"
Output: {{"intent": "set_context", "secondary_intent": "new_search", "secondary_message": "food for 45yo undocumented woman with 2 kids"}}

Message: "My client is a 45yo undocumented woman who speaks Spanish"
Output: {{"intent": "set_context", "secondary_intent": null, "secondary_message": null}}

Message: "ok thanks"
Output: {{"intent": "acknowledge", "secondary_intent": null, "secondary_message": null}}\
"""

_PENDING_ACTION_CONTEXT = """
The assistant previously asked the navigator a follow-up question (pending_action = "{pending_action}").
- If the navigator confirms ("yes", "go ahead", "sure", "do it", "yeah") → intent = "{pending_action}"
- If the navigator declines ("no", "not yet", "later", "never mind", "skip") → intent = "acknowledge"
- If the navigator adds more client info instead → intent = "set_context"
- Otherwise classify the message normally and ignore the pending context.
"""


def _find_previous_human_content(messages: list) -> str | None:
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    if len(human_msgs) >= 2:
        content = human_msgs[-2].content
        return content if isinstance(content, str) else None
    return None


async def resolve_intent_node(state: NavigatorState) -> dict:
    messages = state["messages"]
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        logger.warning("resolve_intent: no human message in state")
        return {"intent": "acknowledge", "intent_queue": [], "secondary_message": None, "pending_action": None}

    pending_action = state.get("pending_action")
    has_groups = bool(state.get("groups"))
    has_results = bool(state.get("results"))

    prior_state = (
        f"{len(state['groups'])} group(s) with results" if has_results
        else f"{len(state['groups'])} group(s), no results yet" if has_groups
        else "no prior search"
    )

    pending_block = (
        _PENDING_ACTION_CONTEXT.format(pending_action=pending_action)
        if pending_action else ""
    )

    system = _SYSTEM_PROMPT.format(
        pending_action_context=pending_block,
        prior_state=prior_state,
    )

    llm = get_llm(settings.classifier_provider, settings.classifier_model, json_mode=True)
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=last_human.content),
    ])

    raw = response.content if isinstance(response.content, str) else "{}"
    logger.info(f"resolve_intent raw: {raw[:300]}")

    try:
        text = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
    except Exception as e:
        logger.error(f"resolve_intent parse error: {e} | raw: {raw[:300]}")
        parsed = {}

    intent = parsed.get("intent", "new_search")
    secondary_intent = parsed.get("secondary_intent")
    secondary_message = parsed.get("secondary_message")

    if intent not in _VALID_INTENTS:
        logger.warning(f"resolve_intent: unknown intent '{intent}', defaulting to new_search")
        intent = "new_search"

    # Fall back if the classified intent requires state that doesn't exist yet
    if intent == "refine" and not has_groups:
        logger.info("resolve_intent: refine with no prior groups → new_search")
        intent = "new_search"
    if intent == "follow_up" and not has_results:
        logger.info("resolve_intent: follow_up with no prior results → new_search")
        intent = "new_search"

    # When confirming a pending action, carry the triggering message forward so
    # classify_groups (Phase 4) can use it instead of the bare "yes"
    if pending_action and intent == pending_action:
        prev_content = _find_previous_human_content(messages)
        if prev_content and prev_content != last_human.content:
            secondary_message = prev_content
            logger.info(f"resolve_intent: confirmed pending_action={pending_action}, secondary_message from history")

    intent_queue = (
        [secondary_intent]
        if secondary_intent and secondary_intent in _VALID_INTENTS and secondary_intent != intent
        else []
    )

    logger.info(
        f"resolve_intent: intent={intent}"
        + (f", queue={intent_queue}" if intent_queue else "")
        + (f", pending_action={pending_action} cleared" if pending_action else "")
    )

    return {
        "intent": intent,
        "intent_queue": intent_queue,
        "secondary_message": secondary_message,
        "pending_action": None,
    }
