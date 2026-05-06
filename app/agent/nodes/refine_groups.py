import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.state import ClientContext, Group, NavigatorState, effective_context
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
- If a per-group client context is provided, factor it into who/what where relevant
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

    case_context = state.get("case_context")
    # Build a per-group view annotated with effective context so the LLM understands
    # who each group is actually for after case + group overrides are merged.
    annotated = []
    for g in existing_groups:
        eff = effective_context(case_context, g.get("client_context"))
        annotated.append({**g, "effective_context": eff})
    existing_str = json.dumps(annotated)
    context_str = _context_summary(case_context)

    llm = get_llm(settings.classifier_provider, settings.classifier_model, json_mode=True)

    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Existing groups: {existing_str}\n"
            f"Case context: {context_str}\n"
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

    # Preserve per-group client_context across refine — the LLM is only allowed to
    # change search params (what/who/where/when), not the person profile attached to a group.
    prior_context_by_id = {g["group_id"]: g.get("client_context") for g in existing_groups}

    groups: list[Group] = []
    for item in groups_data:
        gid = int(item.get("group_id", 1))
        groups.append(Group(
            group_id=gid,
            what=str(item.get("what", "")),
            who=item.get("who") or None,
            where=str(item.get("where") or "San Francisco"),
            when=item.get("when") if item.get("when") not in (None, "null", "") else None,
            open_now=bool(item.get("open_now", False)),
            categories=[],
            eligibilities=[],
            lat=None,
            lng=None,
            client_context=prior_context_by_id.get(gid),
        ))

    groups = [g for g in groups if g["what"]]
    logger.info(f"refine_groups: {len(groups)} refined group(s)")
    return {"groups": groups}
