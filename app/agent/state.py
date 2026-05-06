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


class ClientContext(TypedDict, total=False):
    age: str | None           # e.g. "45yo adult", "senior", "teen"
    housing: str | None       # e.g. "unhoused", "near homeless"
    gender: str | None        # e.g. "woman", "LGBTQ+"
    family_status: str | None # e.g. "single parent, 2 kids"
    employment: str | None    # e.g. "veteran", "unemployed"
    financial: str | None     # e.g. "low-income", "uninsured"
    health: str | None        # e.g. "pregnant", "substance dependency"
    ethnicity: str | None     # e.g. "Latinx", "Filipino/a"
    immigration: str | None   # e.g. "undocumented", "asylum seeker"
    language: str | None      # e.g. "Spanish only", "Cantonese" — not a DB eligibility but affects service fit
    other: str | None         # e.g. "DV survivor", "SF resident"


class NavigatorState(TypedDict):
    messages: Annotated[list, add_messages]
    groups: list[Group]
    results: dict[str, list[dict]]
    formatted: dict[str, dict]  # {group_id: {rationale: str, service_ids: [int]}}
    current_time: str       # sent by frontend, used as fallback for when
    intent: str | None
    client_context: ClientContext | None
    intent_queue: list[str]
    secondary_message: str | None
    pending_action: str | None
