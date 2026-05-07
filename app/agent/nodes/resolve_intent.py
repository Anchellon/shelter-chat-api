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
- refine: A new or narrowed search is needed to answer — even if phrased as a question. Location changes, adding needs, changing eligibility, or asking "do you have info on [service category like shelter/food/jobs] in [location]" all require a new search. **Ordinal references to existing groups ("first group", "second group", "the third group", "the shelter group") always signal refine** — the navigator is modifying, removing, or re-targeting a prior group, not starting over. This holds even if the referenced ordinal doesn't exist (e.g., asking about the "second group" when only one exists) — refine_groups handles that case.
- follow_up: Can be answered from existing results OR from a prior named-org/topic query (see prior_state) without running a new search. Analysis, comparison, ranking, summarizing what was already found, or asking about locations/services from the prior query ("what locations are available?", "what about the one in the Tenderloin?", "list the addresses", "i want to learn more about the one in X").
- query: They're asking about a NEW named org not previously discussed ("what are Glide's hours?", "does Compass accept pets?", "what does the YMCA offer?"). If prior_state already has a prior org/topic query for org X, references to anything WITHIN X — a branch, neighborhood, location, ordinal, or attribute — are `follow_up`, NOT `query`.
- set_context: They're providing or updating client demographics for the case or for a specific group ("my client is a 45yo woman", "new client", "she's also pregnant", "for group 2 the client is a senior", "the family is undocumented"). This includes attributes like age, gender, language, immigration, health, family status — even when scoped to a specific group.
- help: They want to know what the assistant can do OR what categories/types of services are available. Includes capability questions ("what can you do?", "help", "how does this work?") AND category-listing questions ("what's available?", "what types of services exist?", "what kinds of resources are here?", "what categories can you search?", "what services are in SF?"). When in doubt between help and clarify for a vague meta-question with no service mentioned, prefer help.
- acknowledge: Confirming or reacting without requesting action ("ok", "thanks", "got it", "sounds good", "yes" with no context)
- clarify: Message is genuinely too ambiguous to classify — use sparingly, only when truly impossible. **NEVER use clarify if prior_state mentions prior search results or a prior org/topic query** — the message is responding to that context; classify as `follow_up` instead.

Rules:
- Prefer a concrete intent over "clarify" — only resort to clarify if a reasonable guess is impossible
- "refine" only applies when there are prior search groups to modify
- "follow_up" only applies when there are prior search results OR a prior org/topic query to reference
- When prior_state has a prior org/topic query for org X, any reference to a location, branch, ordinal, or item within X is `follow_up` (not `query`, not `clarify`), regardless of how wordy the phrasing is ("I want to learn more about the one in Chinatown", "can you tell me about the Mission branch", "more details on the SoMa one")
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

Message: "Do you have info on shelters in the Tenderloin?"
Output: {{"intent": "refine", "secondary_intent": null, "secondary_message": null}}

Message: "Which of those is closest to 16th and Mission?"
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "What are Glide's current hours?"
Output: {{"intent": "query", "secondary_intent": null, "secondary_message": null}}

Message: "what locations are available?"  (after a prior org/topic query)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "tell me more about the one on Tenderloin"  (after a prior org/topic query)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "Tenderloin"  (terse, after a prior org/topic query about a multi-location org — refers to the Tenderloin branch from the prior list)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "Bayview"  (terse, after a prior org/topic query about a multi-location org)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "the second one"  (after a prior org/topic query — picks an item from the prior list)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "more details"  (after a prior org/topic query)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "i want to learn more about the one in Chinatown"  (after a prior org/topic query about a multi-location org — drill-down, NOT a fresh query)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "can you tell me about the Mission branch?"  (after a prior org/topic query)
Output: {{"intent": "follow_up", "secondary_intent": null, "secondary_message": null}}

Message: "what about Compass?"  (after a prior org/topic query about YMCA — different org, NOT a follow-up)
Output: {{"intent": "query", "secondary_intent": null, "secondary_message": null}}

Message: "I need food resources for a senior"  (after a prior org/topic query — clearly a new search)
Output: {{"intent": "new_search", "secondary_intent": null, "secondary_message": null}}

Message: "My client is a 45yo undocumented woman with 2 kids who needs food"
Output: {{"intent": "set_context", "secondary_intent": "new_search", "secondary_message": "food for 45yo undocumented woman with 2 kids"}}

Message: "My client is a 45yo undocumented woman who speaks Spanish"
Output: {{"intent": "set_context", "secondary_intent": null, "secondary_message": null}}

Message: "for group 2, the client is a 70-year-old senior in a wheelchair"
Output: {{"intent": "set_context", "secondary_intent": null, "secondary_message": null}}

Message: "group 1's client is also pregnant"
Output: {{"intent": "set_context", "secondary_intent": null, "secondary_message": null}}

Message: "actually, the family is undocumented"
Output: {{"intent": "set_context", "secondary_intent": null, "secondary_message": null}}

Message: "she's also a domestic violence survivor"
Output: {{"intent": "set_context", "secondary_intent": null, "secondary_message": null}}

Message: "ok thanks"
Output: {{"intent": "acknowledge", "secondary_intent": null, "secondary_message": null}}

Message: "what is available?"
Output: {{"intent": "help", "secondary_intent": null, "secondary_message": null}}

Message: "what types of services exist in SF?"
Output: {{"intent": "help", "secondary_intent": null, "secondary_message": null}}

Message: "what kinds of services can you search for?"
Output: {{"intent": "help", "secondary_intent": null, "secondary_message": null}}

Message: "what categories do you have?"
Output: {{"intent": "help", "secondary_intent": null, "secondary_message": null}}

Message: "drop the second group"
Output: {{"intent": "refine", "secondary_intent": null, "secondary_message": null}}

Message: "can you search for job resources for the second group?"
Output: {{"intent": "refine", "secondary_intent": null, "secondary_message": null}}

Message: "for the first group, change the location to the Mission"
Output: {{"intent": "refine", "secondary_intent": null, "secondary_message": null}}

Message: "the shelter group should also include food"
Output: {{"intent": "refine", "secondary_intent": null, "secondary_message": null}}\
"""

_PENDING_ACTION_CONTEXT = """
The assistant previously asked the navigator a follow-up question (pending_action = "{pending_action}").
- If the navigator confirms ("yes", "go ahead", "sure", "do it", "yeah") → intent = "{pending_action}"
- If the navigator declines ("no", "not yet", "later", "never mind", "skip") → intent = "acknowledge"
- If the navigator adds more client info instead → intent = "set_context"
- Otherwise classify the message normally and ignore the pending context.
"""

_PENDING_CLARIFY_CONTEXT = """
The assistant previously asked the navigator a clarifying question (search-for-a-client vs general-info).
The navigator's current message is the answer. Classify it as a CONCRETE intent — `new_search`, `query`, `follow_up`, `set_context`, `acknowledge`, or `help`. NEVER return `clarify` again.
- If the answer indicates they want services found for someone (e.g. "specific services", "find services", "for my client", "for my friend", "support services") → `new_search`.
- If the answer asks about a specific named org → `query`.
- Otherwise prefer `new_search` over any other guess.
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
    last_query = state.get("last_query")
    last_query_services = state.get("last_query_services") or []
    has_query_context = bool(last_query_services)

    state_parts: list[str] = []
    if has_results or has_groups:
        group_labels = [
            f"{g.get('what', 'services')} in {g.get('where', 'SF')}"
            for g in state["groups"]
        ]
        label_str = ", ".join(group_labels)
        suffix = "with results" if has_results else "no results yet"
        state_parts.append(f"{len(state['groups'])} group(s) ({label_str}) — {suffix}")
    if has_query_context:
        state_parts.append(
            f"prior org/topic query {last_query!r} ({len(last_query_services)} service(s) available for follow-up)"
        )
    prior_state = "; ".join(state_parts) if state_parts else "no prior search"

    if pending_action == "clarify":
        pending_block = _PENDING_CLARIFY_CONTEXT
    elif pending_action:
        pending_block = _PENDING_ACTION_CONTEXT.format(pending_action=pending_action)
    else:
        pending_block = ""

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
    if intent == "follow_up" and not has_results and not has_query_context:
        logger.info("resolve_intent: follow_up with no prior results or query → new_search")
        intent = "new_search"
    # Drill-down responses to a prior query/results should never hit clarify —
    # the user has already established intent. Force follow_up instead.
    if intent == "clarify" and (has_results or has_query_context):
        logger.info("resolve_intent: clarify with prior state → follow_up")
        intent = "follow_up"
    # If we just asked a clarifying question, the navigator's answer must resolve
    # to a concrete intent — never re-ask. Default to new_search.
    if intent == "clarify" and pending_action == "clarify":
        logger.info("resolve_intent: clarify after pending_action=clarify → new_search")
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
