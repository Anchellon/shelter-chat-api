import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.agent.llm import get_llm
from app.agent.state import ClientContext, Group, NavigatorState, effective_context
from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are updating a client profile for a social services navigator working with a household
or case that may include MULTIPLE PEOPLE. Each search "group" can be for a different person
(e.g. group 1 for the parent, group 2 for the teen, group 3 for the grandmother).

The navigator's message may:
1. SET case-level facts — true of EVERYONE in the household (e.g. "they live in the Mission",
   "they're undocumented", "the family speaks Spanish")
2. SET person-level facts for specific groups — true of one person (e.g. "for group 2 he is 15",
   "the shelter search is for an adult woman", "she's a senior" when only one group exists)
3. UPDATE existing fields
4. CLEAR the entire context — when the navigator says "new client", "clear context", "reset",
   "different client", "next client" — wipes both case-level and per-group context

Decide the SCOPE of the update:
- "case" — applies to all people in this case (or no groups exist yet)
- "groups" — applies to specific named groups; populate target_group_ids
- "ambiguous" — the fact is about a person but NO group is named and 2+ groups exist;
  the user must be asked which group(s) it applies to

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
- If the client clearly needs a specific service (housing, food, shelter, health, jobs, etc.) AND no existing group already covers that need → set pending_action to "new_search"
- If the client clearly needs a specific service AND an existing group already covers that need (same `what` topic, same `where`) → set pending_action to "refine" so the existing group is updated with the new context instead of being replaced
- If the message is a context-only update (no explicit service need) AND it changes a field that affects WHO QUALIFIES for services — gender, immigration, family_status, age, employment, health — AND at least one already-searched group's effective context would change as a result → set pending_action to "refine" so the stale search is re-run with the new eligibility profile.
- Do NOT set pending_action to "refine" for fields that don't affect eligibility filtering (language, ethnicity, financial, housing, other) or for groups marked "not yet searched".
- Otherwise → set pending_action to null

Return ONLY a JSON object. No explanation. No markdown fences.

Format:
{
  "action": "update" | "clear",
  "scope": "case" | "groups" | "ambiguous" | null,
  "target_group_ids": [int] | null,
  "fields": { ...only fields that change... },
  "pending_action": "new_search" | "refine" | null,
  "confirmation": "brief 1-2 sentence confirmation. If pending_action is new_search or refine, end with a confirmation question like 'Want me to search for options?' or 'Want me to update the search?'"
}

Examples:

(no groups exist) Message: "My client is a 45yo undocumented woman who speaks Spanish"
Output: {"action": "update", "scope": "case", "target_group_ids": null, "fields": {"age": "45yo adult", "gender": "woman", "immigration": "undocumented", "language": "Spanish only"}, "pending_action": null, "confirmation": "Got it — 45yo undocumented woman, Spanish-speaking."}

(groups: 1=shelter for adult, 2=youth services for teen) Message: "the family is undocumented"
Output: {"action": "update", "scope": "case", "target_group_ids": null, "fields": {"immigration": "undocumented"}, "pending_action": null, "confirmation": "Got it — added undocumented status for the household."}

(groups: 1=shelter, 2=youth services) Message: "for group 2 he's 15"
Output: {"action": "update", "scope": "groups", "target_group_ids": [2], "fields": {"age": "15", "gender": "man"}, "pending_action": null, "confirmation": "Updated group 2 — 15yo male."}

(groups: 1=shelter, 2=food) Message: "she's a veteran"
Output: {"action": "update", "scope": "ambiguous", "target_group_ids": null, "fields": {"employment": "veteran"}, "pending_action": null, "confirmation": "Got it — adding veteran status."}

(only group 1 exists) Message: "she's a senior"
Output: {"action": "update", "scope": "groups", "target_group_ids": [1], "fields": {"age": "senior"}, "pending_action": null, "confirmation": "Updated — client is a senior."}

Message: "New client"
Output: {"action": "clear", "scope": null, "target_group_ids": null, "fields": {}, "pending_action": null, "confirmation": "Client context cleared. Ready for a new client."}

(groups: 1=shelter for immigrant friend in San Francisco) Message: "he is alone and needs shelter"
Output: {"action": "update", "scope": "groups", "target_group_ids": [1], "fields": {"family_status": "individuals"}, "pending_action": "refine", "confirmation": "Updated — alone, individual. Want me to update the shelter search?"}

(no groups exist) Message: "I need shelter for a single adult male"
Output: {"action": "update", "scope": "case", "target_group_ids": null, "fields": {"gender": "man", "family_status": "individuals"}, "pending_action": "new_search", "confirmation": "Got it — single adult male. Want me to search for shelter options?"}

(groups: 1=shelter for immigrant friend in San Francisco | searched) Message: "hes gay also"
Output: {"action": "update", "scope": "groups", "target_group_ids": [1], "fields": {"gender": "LGBTQ+"}, "pending_action": "refine", "confirmation": "Adding LGBTQ+ status. Want me to update the shelter search with this?"}

(groups: 1=shelter | searched) Message: "she speaks Spanish only"
Output: {"action": "update", "scope": "groups", "target_group_ids": [1], "fields": {"language": "Spanish only"}, "pending_action": null, "confirmation": "Got it — Spanish only."}

(groups: 1=shelter | not yet searched) Message: "shes a senior"
Output: {"action": "update", "scope": "groups", "target_group_ids": [1], "fields": {"age": "senior"}, "pending_action": null, "confirmation": "Got it — senior."}\
"""


def _merge_context(existing: ClientContext | None, fields: dict) -> ClientContext:
    base: ClientContext = dict(existing) if existing else {}  # type: ignore[assignment]
    for key, value in fields.items():
        if value is None:
            base.pop(key, None)  # type: ignore[misc]
        else:
            base[key] = value  # type: ignore[literal-required]
    return base


def _summarise_context(ctx: ClientContext | None) -> str:
    if not ctx:
        return "(none)"
    parts = [f"{k}: {v}" for k, v in ctx.items() if v]
    return ", ".join(parts) if parts else "(none)"


def _build_groups_summary(state: NavigatorState) -> str:
    """Render existing groups + their effective context for the LLM."""
    groups = state.get("groups") or []
    if not groups:
        return "(no groups yet — context updates apply at case level)"
    case = state.get("case_context")
    results = state.get("results") or {}
    lines = []
    for g in groups:
        eff = effective_context(case, g.get("client_context"))
        label = g.get("what", "services")
        if g.get("who"):
            label += f" for {g['who']}"
        # Mark whether this group has produced results yet so the LLM only
        # proposes refine for groups whose searches are actually stale.
        searched = "searched" if results.get(str(g["group_id"])) else "not yet searched"
        lines.append(
            f"- group_id={g['group_id']}: {label} | effective_context: {_summarise_context(eff)} | {searched}"
        )
    return "\n".join(lines)


async def update_client_context_node(state: NavigatorState) -> dict:
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        logger.warning("update_client_context: no human message in state")
        return {}

    case_context = state.get("case_context")
    groups: list[Group] = state.get("groups") or []
    groups_summary = _build_groups_summary(state)
    case_summary = _summarise_context(case_context)

    llm = get_llm(settings.classifier_provider, settings.classifier_model, json_mode=True)

    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Current case context: {case_summary}\n"
            f"Existing groups:\n{groups_summary}\n\n"
            f"Navigator: {last_human.content}"
        )),
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
    scope = parsed.get("scope")
    target_ids = parsed.get("target_group_ids") or []
    fields = parsed.get("fields") or {}
    pending_action = parsed.get("pending_action")
    confirmation = parsed.get("confirmation", "Got it.")

    # ── Clear: wipe case-level AND every per-group context, plus search state ──
    if action == "clear":
        logger.info("update_client_context: clearing context + zeroing search state")
        return {
            "case_context": None,
            "pending_action": pending_action,
            "groups": [],
            "results": {},
            "formatted": {},
            "messages": [AIMessage(content=confirmation)],
        }

    # ── Auto-resolve trivial ambiguity ──
    # 0 groups → must be case-level. 1 group → no real ambiguity, that's the target.
    if scope == "ambiguous":
        if len(groups) == 0:
            scope = "case"
        elif len(groups) == 1:
            scope = "groups"
            target_ids = [groups[0]["group_id"]]

    # ── Genuine ambiguity: interrupt and ask ──
    if scope == "ambiguous":
        clarify_payload = {
            "kind": "context_clarify",
            "proposed_update": fields,
            "groups": [
                {
                    "group_id": g["group_id"],
                    "label": (
                        f"{g.get('what', 'services')}"
                        + (f" for {g['who']}" if g.get("who") else "")
                    ),
                }
                for g in groups
            ],
            "question": (
                "Who does this apply to? Pick one or more groups, or 'everyone' "
                "to apply at the case level."
            ),
        }
        logger.info(f"update_client_context: interrupting for clarify (fields={fields})")
        resume_value = interrupt(clarify_payload)

        if isinstance(resume_value, dict) and resume_value.get("action") == "cancel":
            logger.info("update_client_context: clarify cancelled by user")
            return {
                "messages": [AIMessage(content="Okay, leaving the context as-is.")],
            }

        answers = resume_value.get("answers", {}) if isinstance(resume_value, dict) else {}
        chosen_scope = answers.get("scope")
        chosen_ids = answers.get("group_ids") or []

        if chosen_scope == "case":
            scope = "case"
        else:
            scope = "groups"
            target_ids = [int(i) for i in chosen_ids if i is not None]
            if not target_ids:
                # User submitted nothing — fall back to case so we don't lose the update
                logger.warning("update_client_context: clarify returned no group_ids, applying to case")
                scope = "case"

    # ── Apply the update ──
    if scope == "case" or not groups:
        new_case = _merge_context(case_context, fields)
        logger.info(f"update_client_context: case-level update → {new_case}")
        return {
            "case_context": new_case,
            "pending_action": pending_action,
            "messages": [AIMessage(content=confirmation)],
        }

    # scope == "groups"
    target_set = set(target_ids)
    updated_groups: list[Group] = []
    matched = 0
    for g in groups:
        if g["group_id"] in target_set:
            new_ctx = _merge_context(g.get("client_context"), fields)
            updated_groups.append({**g, "client_context": new_ctx})  # type: ignore[misc]
            matched += 1
        else:
            updated_groups.append(g)

    if matched == 0:
        # LLM picked groups that don't exist — fall back to case so we don't lose the update
        logger.warning(f"update_client_context: target_ids {target_ids} don't match any group, applying to case")
        new_case = _merge_context(case_context, fields)
        return {
            "case_context": new_case,
            "pending_action": pending_action,
            "messages": [AIMessage(content=confirmation)],
        }

    logger.info(f"update_client_context: per-group update on {matched} group(s) → fields={fields}")
    return {
        "groups": updated_groups,
        "pending_action": pending_action,
        "messages": [AIMessage(content=confirmation)],
    }
