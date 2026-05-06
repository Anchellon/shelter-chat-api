import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.state import ClientContext, Group, NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are refining an existing social services search for a navigator.

The navigator has prior search groups and wants to modify them. Apply their change and return the updated groups list.

You can:
- Modify when/open_now/where/who/what on existing groups
- Add entirely new groups
- Remove groups the navigator no longer wants

Rules:
- Preserve group_id for groups that remain
- New groups get group_id = max existing + 1, 2, etc.
- To remove a group, simply omit it from the output
- Keep all fields — reset categories/eligibilities/lat/lng to empty (intake will re-populate them)
- If client_context is provided, factor it into who/what where relevant
- If no location is mentioned in the change, keep the existing where value
- Set open_now to true ONLY if the navigator explicitly asks for open services

Return ONLY a JSON object with a "groups" key. No explanation. No markdown fences.

Format:
{"groups": [{"group_id": 1, "what": "...", "who": "...", "where": "...", "when": null, "open_now": false, "categories": [], "eligibilities": [], "lat": null, "lng": null}]}

Examples:

Existing: [{"group_id": 1, "what": "shelter", "who": "adult male", "where": "San Francisco", "when": null, "open_now": false}]
Change: "same but only open now"
Output: {"groups": [{"group_id": 1, "what": "shelter", "who": "adult male", "where": "San Francisco", "when": null, "open_now": true, "categories": [], "eligibilities": [], "lat": null, "lng": null}]}

Existing: [{"group_id": 1, "what": "shelter", "who": "adult", "where": "San Francisco", "when": null, "open_now": false}]
Change: "actually she's a senior"
Output: {"groups": [{"group_id": 1, "what": "shelter", "who": "senior woman", "where": "San Francisco", "when": null, "open_now": false, "categories": [], "eligibilities": [], "lat": null, "lng": null}]}\
"""


def _context_summary(context: ClientContext | None) -> str:
    if not context:
        return "None"
    parts = []
    for key, val in context.items():
        if val:
            parts.append(f"{key}: {val}")
    return ", ".join(parts) if parts else "None"


async def refine_groups_node(state: NavigatorState) -> dict:
    existing_groups = state.get("groups") or []
    messages = state["messages"]

    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        logger.warning("refine_groups: no human message in state")
        return {}

    if not existing_groups:
        logger.warning("refine_groups: no existing groups to refine — returning empty")
        return {"groups": []}

    client_context = state.get("client_context")
    context_str = _context_summary(client_context)
    existing_str = json.dumps(existing_groups)

    llm = get_llm(settings.classifier_provider, settings.classifier_model, json_mode=True)

    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Existing groups: {existing_str}\n"
            f"Client context: {context_str}\n"
            f"Navigator change: {last_human.content}"
        )),
    ])

    raw = response.content if isinstance(response.content, str) else "{}"
    logger.info(f"refine_groups raw: {raw[:300]}")

    try:
        text = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        groups_data = parsed.get("groups", [])
    except Exception as e:
        logger.error(f"refine_groups parse error: {e} | raw: {raw[:300]}")
        return {"groups": existing_groups}

    groups: list[Group] = []
    for item in groups_data:
        groups.append(Group(
            group_id=int(item.get("group_id", 1)),
            what=str(item.get("what", "")),
            who=item.get("who") or None,
            where=str(item.get("where") or "San Francisco"),
            when=item.get("when") if item.get("when") not in (None, "null", "") else None,
            open_now=bool(item.get("open_now", False)),
            categories=[],
            eligibilities=[],
            lat=None,
            lng=None,
        ))

    groups = [g for g in groups if g["what"]]
    logger.info(f"refine_groups: {len(groups)} refined group(s)")
    return {"groups": groups}
