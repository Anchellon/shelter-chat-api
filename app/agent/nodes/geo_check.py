import json
import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.state import Group, NavigatorState

logger = logging.getLogger(__name__)

_SF_LAT_MIN = 37.63
_SF_LAT_MAX = 37.84
_SF_LNG_MIN = -122.52
_SF_LNG_MAX = -122.35


def _parse_geo(result: Any) -> tuple[float | None, float | None]:
    if isinstance(result, dict):
        return result.get("lat"), result.get("lng")
    return None, None


def _is_outside_sf(lat: float | None, lng: float | None) -> bool:
    if lat is None or lng is None:
        return False  # geocoder couldn't resolve — let it pass
    return not (_SF_LAT_MIN <= lat <= _SF_LAT_MAX and _SF_LNG_MIN <= lng <= _SF_LNG_MAX)


def _unwrap_tool_result(result: Any) -> Any:
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        text = result[0]["text"]
        try:
            return json.loads(text)
        except Exception:
            return text
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return result
    return result


def build_geo_check_node(tools_by_name: dict):
    """
    Factory — returns a plain (non-interruptible) node that geocodes each group
    and validates it falls within San Francisco. Groups outside SF are dropped;
    if ALL groups are outside SF an AIMessage refusal is added to state and the
    graph routes to END via after_geo_check in graph.py.
    """
    async def geo_check_node(state: NavigatorState) -> dict:
        groups = state["groups"]
        geo_tool = tools_by_name.get("geocode_location")

        updated_groups: list[Group] = []
        rejected_locations: list[str] = []

        for group in groups:
            lat, lng = None, None
            geocoding_attempted = False
            if geo_tool and group.get("where"):
                geocoding_attempted = True
                geo_result = _unwrap_tool_result(
                    await geo_tool.ainvoke({"location_text": group["where"]})
                )
                lat, lng = _parse_geo(geo_result)

            if geocoding_attempted and lat is None and lng is None:
                logger.info(
                    f"geo_check: group {group['group_id']} '{group['where']}'"
                    f" could not be geocoded — treating as outside SF"
                )
                rejected_locations.append(group["where"])
                continue

            if _is_outside_sf(lat, lng):
                logger.info(
                    f"geo_check: group {group['group_id']} '{group['where']}'"
                    f" ({lat},{lng}) is outside SF — skipping"
                )
                rejected_locations.append(group["where"])
                continue

            updated_groups.append(Group(
                group_id=group["group_id"],
                what=group["what"],
                who=group["who"],
                where=group["where"],
                when=group["when"],
                open_now=group.get("open_now", False),
                categories=group.get("categories", []),
                eligibilities=group.get("eligibilities", []),
                lat=lat,
                lng=lng,
            ))

        if rejected_locations and not updated_groups:
            loc = (
                rejected_locations[0]
                if len(rejected_locations) == 1
                else " and ".join(rejected_locations)
            )
            logger.info(f"geo_check: all groups outside SF ({loc}) — returning refusal")
            return {
                "groups": [],
                "messages": [AIMessage(
                    content=(
                        f"I can only find services in San Francisco. "
                        f"{loc} is outside San Francisco — please describe "
                        f"your needs for a San Francisco location."
                    )
                )],
            }

        logger.info(f"geo_check: {len(updated_groups)} group(s) within SF bounds")
        return {"groups": updated_groups}

    return geo_check_node
