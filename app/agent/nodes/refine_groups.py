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
- Output only the editable search fields (group_id, what, who, where, when, open_now). Downstream code preserves categories/eligibilities/lat/lng on groups whose what/who/where you didn't change.
- If a per-group client context is provided, factor it into who/what where relevant
- If no location is mentioned in the change, keep the existing where value
- Set open_now to true ONLY if the navigator explicitly asks for open services

Return ONLY a JSON object with a "groups" key. No explanation. No markdown fences.

Format:
{"groups": [{"group_id": 1, "what": "...", "who": "...", "where": "...", "when": null, "open_now": false}]}

Examples:

Existing: [{"group_id": 1, "what": "shelter", "who": "adult male", "where": "San Francisco", "when": null, "open_now": false}]
Change: "same but only open now"
Output: {"groups": [{"group_id": 1, "what": "shelter", "who": "adult male", "where": "San Francisco", "when": null, "open_now": true}]}

Existing: [{"group_id": 1, "what": "shelter", "who": "adult", "where": "San Francisco", "when": null, "open_now": false}]
Change: "actually she's a senior"
Output: {"groups": [{"group_id": 1, "what": "shelter", "who": "senior woman", "where": "San Francisco", "when": null, "open_now": false}]}

Existing: [{"group_id": 1, "what": "shelter", "who": null, "where": "Larkin St", "when": null, "open_now": false}, {"group_id": 2, "what": "health", "who": null, "where": "Larkin St", "when": null, "open_now": false}]
Change: "find job resources for the second group"
Output: {"groups": [{"group_id": 1, "what": "shelter", "who": null, "where": "Larkin St", "when": null, "open_now": false}, {"group_id": 2, "what": "health and job resources", "who": null, "where": "Larkin St", "when": null, "open_now": false}]}\
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

    # Preserve intake-derived fields on groups whose search params didn't change.
    # This avoids re-running intake/geocoding on groups the navigator didn't touch.
    prior_by_id = {g["group_id"]: g for g in existing_groups}

    groups: list[Group] = []
    changed_group_ids: list[int] = []
    for item in groups_data:
        gid = int(item.get("group_id", 1))
        new_what = str(item.get("what", ""))
        new_who = item.get("who") or None
        new_where = str(item.get("where") or "San Francisco")
        new_when = item.get("when") if item.get("when") not in (None, "null", "") else None
        new_open_now = bool(item.get("open_now", False))

        prior = prior_by_id.get(gid)
        if prior is not None:
            # categories derive from `what`, eligibilities from `who`, lat/lng from `where`.
            # Preserve each only if the source field is unchanged.
            keep_categories = prior.get("categories") or [] if prior.get("what") == new_what else []
            keep_eligibilities = prior.get("eligibilities") or [] if prior.get("who") == new_who else []
            if prior.get("where") == new_where:
                keep_lat = prior.get("lat")
                keep_lng = prior.get("lng")
            else:
                keep_lat = None
                keep_lng = None
            keep_client_context = prior.get("client_context")
            is_changed = (
                prior.get("what") != new_what
                or prior.get("who") != new_who
                or prior.get("where") != new_where
                or prior.get("when") != new_when
                or bool(prior.get("open_now", False)) != new_open_now
            )
        else:
            keep_categories = []
            keep_eligibilities = []
            keep_lat = None
            keep_lng = None
            keep_client_context = None
            is_changed = True  # newly added group

        if is_changed and new_what:
            changed_group_ids.append(gid)

        groups.append(Group(
            group_id=gid,
            what=new_what,
            who=new_who,
            where=new_where,
            when=new_when,
            open_now=new_open_now,
            categories=keep_categories,
            eligibilities=keep_eligibilities,
            lat=keep_lat,
            lng=keep_lng,
            client_context=keep_client_context,
        ))

    groups = [g for g in groups if g["what"]]
    new_ids = {g["group_id"] for g in groups}
    removed_group_ids = [gid for gid in prior_by_id if gid not in new_ids]
    logger.info(
        "refine_groups: %d refined group(s); changed=%s; removed=%s; preserved=%s",
        len(groups),
        changed_group_ids,
        removed_group_ids,
        [
            {
                "id": g["group_id"],
                "cats": bool(g["categories"]),
                "elig": bool(g["eligibilities"]),
                "geo": g["lat"] is not None,
            }
            for g in groups
        ],
    )
    return {
        "groups": groups,
        "changed_group_ids": changed_group_ids,
        "removed_group_ids": removed_group_ids,
    }
