from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class Group(TypedDict):
    group_id: int
    what: str
    who: str | None
    where: str
    when: str | None        # e.g. "Saturday morning" — null if not mentioned
    open_now: bool          # True only if user explicitly asks for open services
    # Populated by intake
    categories: list[str]   # mapped from what via list_categories (can be multiple)
    eligibilities: list     # mapped from who via list_eligibilities
    lat: float | None       # from geocode_location
    lng: float | None


class NavigatorState(TypedDict):
    messages: Annotated[list, add_messages]
    groups: list[Group]
    results: dict[str, list[dict]]
    formatted: dict[str, dict]  # {group_id: {rationale: str, service_ids: [int]}}
    current_time: str       # sent by frontend, used as fallback for when
