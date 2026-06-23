import json
import logging
import time

from app.agent.state import NavigatorState
from app.core.metrics import record_mcp_search_duration

logger = logging.getLogger(__name__)

# Matches services[:5] in format_results._rationale_for_group and
# converse._format_results_summary — the only downstream consumers of state["results"].
_ENRICH_TOP_N_PER_GROUP = 5


def _parse_tool_result(raw) -> list:
    """Unwrap MCP-style {type: text, text: ...} payloads into a Python list."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        try:
            raw = json.loads(raw[0]["text"])
        except Exception:
            return []
    return raw if isinstance(raw, list) else []


async def _enrich_top_results(
    results: dict[str, list[dict]],
    tools_by_name: dict,
) -> None:
    """Backfill name/org_name/address/phone onto the top services per group.

    search_services returns filter/search fields (service_id, lat/lng, schedule,
    category_names, eligibility_*, embedding_text) but not name, org_name, or
    address. Downstream consumers — format_results' rationale generator and
    _format_results_summary's follow-up prompt — read those fields and silently
    render "Unknown" when they're absent. We batch-fetch full details for the
    top N services per group via get_service_details_batch and merge in place.

    Mirrors the same enrichment we do for query results in converse.py.
    """
    batch_tool = tools_by_name.get("get_service_details_batch")
    if batch_tool is None:
        return

    ids_to_fetch: list[int] = []
    for services in results.values():
        for svc in services[:_ENRICH_TOP_N_PER_GROUP]:
            sid = svc.get("service_id")
            if sid is not None and not svc.get("name"):
                ids_to_fetch.append(sid)

    if not ids_to_fetch:
        return

    try:
        raw = await batch_tool.ainvoke({"service_ids": ids_to_fetch})
        details = _parse_tool_result(raw)
        details_by_id = {
            d.get("service_id"): d for d in details if isinstance(d, dict) and d.get("service_id") is not None
        }
        for services in results.values():
            for svc in services:
                detail = details_by_id.get(svc.get("service_id"))
                if not detail:
                    continue
                for k, v in detail.items():
                    if v not in (None, "", [], {}):
                        svc[k] = v
        logger.info(f"search_per_group: enriched {len(details_by_id)} service(s) via batch detail fetch")
    except Exception as e:
        logger.warning(f"search_per_group: enrichment failed: {e}")


def build_search_per_group_node(tools_by_name: dict):
    """
    Factory — returns the search_per_group_node function with MCP tools in closure.
    """
    search_tool = tools_by_name.get("search_services")

    async def search_per_group_node(state: NavigatorState) -> dict:
        groups = state["groups"]
        results: dict[str, list[dict]] = {}

        if not search_tool:
            logger.warning("search_per_group: search_services tool not available")
            return {"results": results}

        for group in groups:
            group_id = str(group["group_id"])
            query_parts = [group["what"]]
            if group.get("who"):
                query_parts.append(f"for {group['who']}")
            if group.get("where"):
                query_parts.append(f"in {group['where']}")
            args = {
                "query": " ".join(query_parts),
            }
            if group.get("categories"):
                args["categories"] = group["categories"]
            if group.get("eligibilities"):
                args["eligibilities"] = group["eligibilities"]
            if group.get("lat") is not None:
                args["lat"] = group["lat"]
            if group.get("lng") is not None:
                args["lng"] = group["lng"]
            if group.get("open_now") and group.get("when"):
                args["when"] = group["when"]

            logger.info(f"search_per_group: group_id={group_id}, args={args}")
            _t0 = time.monotonic()
            try:
                raw = await search_tool.ainvoke(args)
                record_mcp_search_duration((time.monotonic() - _t0) * 1000, "success")
            except Exception:
                record_mcp_search_duration((time.monotonic() - _t0) * 1000, "error")
                raise
            results[group_id] = _parse_tool_result(raw)
            logger.info(f"search_per_group: group_id={group_id} → {len(results[group_id])} results")

        await _enrich_top_results(results, tools_by_name)

        return {"results": results}

    return search_per_group_node
