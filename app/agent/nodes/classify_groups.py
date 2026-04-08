import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.state import Group, NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a query classifier. Extract need groups from the user's message.

Rules:
- Only extract NEED groups. A need is something the user or someone they know is LOOKING FOR or REQUIRES.
- OFFERS are NOT needs. If the message says "I can provide X", "I offer X", "I am giving X", "I have X available" — that is an offer. Do NOT include it. Return {"groups": []} for offers.
- Group by WHO + WHERE. Each unique combination of who + where = one group. If multiple needs share the same "who" and "where", combine them into ONE group with a combined "what" (e.g. "food and shelter"). Only create separate groups when "who" or "where" differ.
- A query can produce multiple groups (e.g. different populations or locations). Each group can represent multiple service needs and a population with multiple characteristics — capture all of them in "what" and "who" as natural language.
- If no location is mentioned, set "where" to "San Francisco".
- Copy "what", "who", and "when" as raw natural language — do not normalize or categorize.
- Set "who" to null if no specific population is mentioned.
- Set "when" to null if no time or day is mentioned. Extract it if the user says things like "tonight", "Saturday morning", "after 6pm", "on weekdays", etc.
- Set "open_now" to true ONLY if the user explicitly wants open services right now — phrases like "open now", "currently open", "open today", "what's open". Set to false otherwise, even if "when" is mentioned.
- group_id starts at 1 and increments per group.

Return ONLY a JSON object with a "groups" key. No explanation. No markdown fences.

Examples:

User: "I need shelter and food for an adult on eddy street"
Output: {"groups": [{"group_id": 1, "what": "shelter and food", "who": "adult", "where": "Eddy Street, San Francisco", "when": null, "open_now": false}]}

User: "I have a group of lgbtq teens who need food and shelter"
Output: {"groups": [{"group_id": 1, "what": "food and shelter", "who": "lgbtq teens", "where": "San Francisco", "when": null, "open_now": false}]}

User: "I need shelter for seniors and food for my kids on Saturday morning"
Output: {"groups": [{"group_id": 1, "what": "shelter", "who": "seniors", "where": "San Francisco", "when": "Saturday morning", "open_now": false}, {"group_id": 2, "what": "food", "who": "kids", "where": "San Francisco", "when": "Saturday morning", "open_now": false}]}

User: "What shelters are open now in the Tenderloin?"
Output: {"groups": [{"group_id": 1, "what": "shelter", "who": null, "where": "Tenderloin, San Francisco", "when": null, "open_now": true}]}

User: "I can offer free meals to anyone who needs them"
Output: {"groups": []}

User: "Looking for drug rehab in the Tenderloin"
Output: {"groups": [{"group_id": 1, "what": "drug rehab", "who": null, "where": "Tenderloin, San Francisco", "when": null, "open_now": false}]}"""


def _parse_groups(raw: str) -> list[Group]:
    """Extract a JSON array (or single object) from the LLM response."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", raw).strip()

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    array_match = re.search(r"\[.*\]", text, re.DOTALL)

    if object_match:
        parsed = json.loads(object_match.group())
        # Preferred: {"groups": [...]}
        if isinstance(parsed, dict) and "groups" in parsed:
            data = parsed["groups"] if isinstance(parsed["groups"], list) else []
        else:
            data = [parsed]
    elif array_match:
        data = json.loads(array_match.group())
        if not isinstance(data, list):
            data = [data]
    else:
        raise ValueError(f"No JSON found in response: {raw[:200]}")

    groups: list[Group] = []
    for i, item in enumerate(data, start=1):
        groups.append(Group(
            group_id=int(item.get("group_id", i)),
            what=str(item.get("what", "")),
            who=item.get("who") or None,
            where=str(item.get("where") or "San Francisco"),
            when=item.get("when") if item.get("when") not in (None, "null", "") else None,
            open_now=bool(item.get("open_now", False)),
            # Populated later by intake
            categories=[],
            eligibilities=[],
            lat=None,
            lng=None,
        ))
    return [g for g in groups if g["what"]]


async def classify_groups_node(state: NavigatorState) -> dict:
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human is None:
        logger.warning("classify_groups: no human message in state")
        return {"groups": []}

    user_text = last_human.content if isinstance(last_human.content, str) else ""

    llm = get_llm(settings.classifier_provider, settings.classifier_model, json_mode=True)

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ])

    raw = response.content if isinstance(response.content, str) else ""
    logger.info(f"classify_groups raw response: {raw[:300]}")

    try:
        groups = _parse_groups(raw)
    except Exception as e:
        logger.error(f"classify_groups parse error: {e} | raw: {raw[:300]}")
        groups = []

    logger.info(f"classify_groups extracted {len(groups)} group(s): {groups}")

    return {"groups": groups}
