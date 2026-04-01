import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.state import NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are summarizing search results for a social services navigator.
Write exactly 1-2 sentences explaining why the listed services were selected for this need.
Focus on how the services match the need (what, who, where). Be concise and warm.
Do not list service names or invent details not in the provided text.
Respond with only the rationale sentences — no prefixes, no labels."""


def build_format_results_node():
    llm = get_llm(settings.formatter_provider, settings.formatter_model)

    async def _rationale_for_group(group: dict, services: list) -> str:
        if not services:
            return f"No services found for {group['what']} in {group['where']}."

        snippets = []
        for svc in services[:5]:
            text = svc.get("embedding_text") or ""
            if text:
                snippets.append(text[:300])

        prompt = (
            f"Need: {group['what']}"
            + (f", for: {group['who']}" if group.get("who") else "")
            + f", near: {group['where']}\n\n"
            + "Services found:\n"
            + "\n---\n".join(snippets)
        )

        response = await llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content.strip() if isinstance(response.content, str) else ""

    async def format_results_node(state: NavigatorState) -> dict:
        groups = state["groups"]
        results = state.get("results") or {}
        formatted: dict[str, dict] = {}

        for group in groups:
            gid = str(group["group_id"])
            services = results.get(gid, [])
            service_ids = [s["service_id"] for s in services]
            rationale = await _rationale_for_group(group, services)
            formatted[gid] = {"rationale": rationale, "service_ids": service_ids}
            logger.info(f"format_results: group {gid} → {len(service_ids)} services")

        summary_lines = []
        for group in groups:
            gid = str(group["group_id"])
            rationale = formatted[gid].get("rationale", "")
            count = len(formatted[gid].get("service_ids", []))
            summary_lines.append(f"{rationale} ({count} result{'s' if count != 1 else ''} found)")

        return {
            "formatted": formatted,
            "messages": [AIMessage(content="\n\n".join(summary_lines))],
        }

    return format_results_node
