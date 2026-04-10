import logging

from app.agent.state import NavigatorState

logger = logging.getLogger(__name__)


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
            raw = await search_tool.ainvoke(args)

            # MCP tool results may come back as [{"type": "text", "text": "..."}]
            if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
                import json
                try:
                    raw = json.loads(raw[0]["text"])
                except Exception:
                    raw = []
            if not isinstance(raw, list):
                raw = []

            results[group_id] = raw
            logger.info(f"search_per_group: group_id={group_id} → {len(raw)} results")

        return {"results": results}

    return search_per_group_node
