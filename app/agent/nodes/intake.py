import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from app.agent.llm import get_llm
from app.agent.state import Group, NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

_MAP_CATEGORY_PROMPT = """\
Map the service need to all matching categories from the list below.
Return a JSON object with a "categories" key containing an array of matching category names exactly as written.

Examples:
Need: emergency shelter -> {{"categories": ["sfsg-shelter"]}}
Need: food bank -> {{"categories": ["sfsg-food"]}}
Need: drug rehab -> {{"categories": ["sfsg-substanceuse", "sfsg-health"]}}
Need: job training -> {{"categories": ["sfsg-jobs"]}}
Need: nothing relevant -> {{"categories": []}}

Categories:
{categories}

Need: {what}
Output:"""

_MAP_ELIGIBILITY_PROMPT = """\
Map who needs this service to matching values from the list below.
Return a JSON object with an "eligibilities" key containing an array of matching strings.

Examples:
Who: homeless veteran -> {{"eligibilities": ["Adults", "Veterans", "Experiencing Homelessness"]}}
Who: pregnant teenager -> {{"eligibilities": ["Teens", "Pregnant"]}}
Who: single adult -> {{"eligibilities": ["Adults", "Individuals"]}}
Who: family with kids -> {{"eligibilities": ["Families with children below 18 years old"]}}
Who: undocumented immigrant -> {{"eligibilities": ["Immigrants", "Undocumented"]}}
Who: senior citizen -> {{"eligibilities": ["Senior"]}}
Who: low income woman -> {{"eligibilities": ["Women", "Low-Income"]}}
Who: anyone -> {{"eligibilities": ["Anyone in Need"]}}
Who: LGBTQ youth -> {{"eligibilities": ["Teens", "Transitional Aged Youth (TAY)", "LGBTQ+"]}}
Who: infant or toddler -> {{"eligibilities": ["Infants", "Toddlers"]}}
Who: child -> {{"eligibilities": ["Children"]}}
Who: kids -> {{"eligibilities": ["Children"]}}
Who: someone in jail -> {{"eligibilities": ["In Jail"]}}
Who: nearly homeless person -> {{"eligibilities": ["Near Homeless"]}}
Who: homeowner -> {{"eligibilities": ["Home Owners"]}}
Who: renter -> {{"eligibilities": ["Home Renters"]}}
Who: man -> {{"eligibilities": ["Men"]}}
Who: married couple without kids -> {{"eligibilities": ["Married no children"]}}
Who: single parent -> {{"eligibilities": ["Single Parent"]}}
Who: employed person -> {{"eligibilities": ["Employed"]}}
Who: retired person -> {{"eligibilities": ["Retired"]}}
Who: unemployed person -> {{"eligibilities": ["Unemployed"]}}
Who: uninsured person -> {{"eligibilities": ["Uninsured"]}}
Who: person with HIV -> {{"eligibilities": ["HIV/AIDS"]}}
Who: person with disability -> {{"eligibilities": ["Special Needs/Disabilities"]}}
Who: person with substance abuse -> {{"eligibilities": ["Substance Dependency"]}}
Who: visually impaired person -> {{"eligibilities": ["Visual Impairment"]}}
Who: deaf person -> {{"eligibilities": ["Deaf or Hard of Hearing"]}}
Who: Black person -> {{"eligibilities": ["African/Black"]}}
Who: Asian person -> {{"eligibilities": ["API (Asian/Pacific Islander)"]}}
Who: Chinese person -> {{"eligibilities": ["Chinese"]}}
Who: Filipino person -> {{"eligibilities": ["Filipino/a"]}}
Who: Jewish person -> {{"eligibilities": ["Jewish"]}}
Who: Latino person -> {{"eligibilities": ["Latinx"]}}
Who: Middle Eastern person -> {{"eligibilities": ["Middle Eastern and North African"]}}
Who: Native American -> {{"eligibilities": ["Native American"]}}
Who: Pacific Islander -> {{"eligibilities": ["Pacific Islander"]}}
Who: Samoan -> {{"eligibilities": ["Samoan"]}}
Who: domestic violence survivor -> {{"eligibilities": ["Domestic Violence Survivors", "Gender-Based Violence"]}}
Who: human trafficking survivor -> {{"eligibilities": ["Human Trafficking Survivors"]}}
Who: SF resident -> {{"eligibilities": ["San Francisco Residents"]}}
Who: sexual assault survivor -> {{"eligibilities": ["Sexual Assault Survivors"]}}
Who: trauma survivor -> {{"eligibilities": ["Trauma Survivors"]}}
Who: abuse survivor -> {{"eligibilities": ["Abuse or Neglect Survivors"]}}
Who: disaster victim -> {{"eligibilities": ["Disaster Victim"]}}

Available values:
{eligibilities}

Who: {who}
Output:"""



def _unwrap_tool_result(result: Any) -> Any:
    """
    MCP tools called via LangChain adapter may return:
      - the raw Python value (list, dict, str)
      - a JSON string
      - a list of content dicts [{"type": "text", "text": "..."}]
    Normalise to the actual value.
    """
    # List of content dicts from MCP — extract the text and parse it
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        text = result[0]["text"]
        try:
            return json.loads(text)
        except Exception:
            return text
    # Raw JSON string
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return result
    return result


def build_intake_node(tools_by_name: dict):
    """
    Factory — returns the intake_node function with MCP tools in closure.
    Called once during graph construction.
    """
    llm_json = get_llm(settings.intake_provider, settings.intake_model, json_mode=True, max_tokens=256)

    async def _map_categories(what: str, categories: list[str]) -> list[str]:
        prompt = _MAP_CATEGORY_PROMPT.format(
            categories="\n".join(f"- {c}" for c in categories),
            what=what,
        )
        response = await llm_json.ainvoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else "{}"
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                result = result.get("categories", [])
            if isinstance(result, list):
                return [c for c in result if isinstance(c, str) and c in categories]
        except Exception:
            pass
        return [c for c in categories if c in raw]

    async def _map_eligibilities(who: str, eligibilities: dict) -> list[str]:
        all_values = [v for vals in eligibilities.values() for v in vals]
        prompt = _MAP_ELIGIBILITY_PROMPT.format(
            eligibilities="\n".join(f"- {v}" for v in all_values),
            who=who,
        )
        response = await llm_json.ainvoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else "{}"
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                result = result.get("eligibilities", [])
            if isinstance(result, list):
                return [v for v in result if isinstance(v, str) and v in all_values]
        except Exception:
            pass
        # Last resort: scan raw string for known values
        return [v for v in all_values if v in raw]

    async def intake_node(state: NavigatorState) -> dict:
        groups = state["groups"]
        current_time = state.get("current_time") or ""

        cats_tool = tools_by_name.get("list_categories")
        eligs_tool = tools_by_name.get("list_eligibilities")

        categories: list[str] = _unwrap_tool_result(await cats_tool.ainvoke({})) if cats_tool else []
        eligibilities: dict = _unwrap_tool_result(await eligs_tool.ainvoke({})) if eligs_tool else {}

        updated_groups = []

        for group in groups:
            # --- Mapping ---
            mapped_cats = await _map_categories(group["what"], categories)
            mapped_elig = await _map_eligibilities(group["who"], eligibilities) if group["who"] else []
            if "Anyone in Need" not in mapped_elig:
                mapped_elig.append("Anyone in Need")

            # --- Gap detection ---
            gaps = []
            if not mapped_cats:
                gaps.append({
                    "dimension": "what",
                    "type": "multi_select",
                    "question": "What type of service are they looking for?",
                    "options": categories,
                })
            if not group["who"]:
                gaps.append({
                    "dimension": "who",
                    "type": "multi_select",
                    "question": "Who is this for?",
                    "options": eligibilities,
                })

            # --- HITL interrupt if gaps ---
            if gaps:
                logger.info(f"intake: group {group['group_id']} has gaps {[g['dimension'] for g in gaps]}, interrupting")
                response = interrupt({
                    "group_id": group["group_id"],
                    "group_label": f"Group {group['group_id']} · {group['what']}",
                    "steps": gaps,
                })

                if isinstance(response, dict) and response.get("action") == "cancel":
                    logger.info("intake: cancelled by user")
                    return {"groups": []}

                answers = response.get("answers", {}) if isinstance(response, dict) else {}

                if "what" in answers:
                    what_ans = answers["what"]
                    mapped_cats = what_ans if isinstance(what_ans, list) else [what_ans]
                if "who" in answers:
                    who_ans = answers["who"]
                    mapped_elig = who_ans if isinstance(who_ans, list) else [who_ans]
                    if "Anyone in Need" not in mapped_elig:
                        mapped_elig.append("Anyone in Need")

            updated_groups.append(Group(
                group_id=group["group_id"],
                what=group["what"],
                who=group["who"],
                where=group["where"],
                when=group["when"] or current_time,
                open_now=group.get("open_now", False),
                categories=mapped_cats,
                eligibilities=mapped_elig,
                lat=group.get("lat"),
                lng=group.get("lng"),
            ))

        logger.info(f"intake: {len(updated_groups)} group(s) complete")
        return {"groups": updated_groups}

    return intake_node
