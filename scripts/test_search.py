"""
Smoke test for the full pipeline: classify_groups → intake → search_per_group → agent.
Requires: Ollama + MCP server both running.

Run from project root:
    python -m scripts.test_search
"""
import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agent.graph import build_graph
from app.core.config import settings


async def run():
    print("Connecting to MCP server...")
    client = MultiServerMCPClient({
        "shelter": {"url": settings.mcp_server_url, "transport": "streamable_http"}
    })
    tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}\n")

    graph = build_graph(tools, MemorySaver())
    config = {"configurable": {"thread_id": "search-test-1"}}
    query = "Looking for emergency shelter in the Tenderloin"

    print(f"QUERY: {query}")
    print("=" * 60)

    async for event in graph.astream_events(
        {
            "messages": [HumanMessage(content=query)],
            "current_time": "Monday 14:00",
            "groups": [],
            "results": {},
            "formatted": {},
        },
        config=config,
        version="v2",
    ):
        name = event.get("name", "")
        kind = event["event"]

        if kind == "on_chain_end" and name == "classify_groups":
            groups = event.get("data", {}).get("output", {}).get("groups", [])
            print(f"classify_groups → {json.dumps(groups, indent=2)}\n")

        elif kind == "on_chain_end" and name == "search_per_group":
            results = event.get("data", {}).get("output", {}).get("results", {})
            for gid, services in results.items():
                print(f"search_per_group group {gid} → {len(services)} result(s)")
                for svc in services[:3]:
                    dist = f"  {svc.get('distance_km')} km" if svc.get('distance_km') is not None else ""
                    print(f"  service_id={svc.get('service_id')} cat={svc.get('sfsg_category_names')}{dist}")
            print()

        elif kind == "on_chain_end" and name == "format_results":
            formatted = event.get("data", {}).get("output", {}).get("formatted", {})
            for gid, data in formatted.items():
                print(f"\nformat_results group {gid}:")
                print(f"  service_ids: {data.get('service_ids')}")
                print(f"  rationale: {data.get('rationale')}")
            print()

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk.content, str) and chunk.content:
                print(chunk.content, end="", flush=True)

    # Check for pending interrupt (intake gap)
    state = await graph.aget_state(config)
    interrupt_data = None
    for task in state.tasks:
        for intr in getattr(task, "interrupts", []):
            interrupt_data = intr.value if hasattr(intr, "value") else intr

    if interrupt_data:
        print("\n\nINTERRUPT — answering gaps automatically...")
        answers = {}
        for step in interrupt_data.get("steps", []):
            dim = step["dimension"]
            opts = step["options"]
            if step["type"] == "single_select":
                answers[dim] = opts[0] if opts else ""
            elif step["type"] == "multi_select":
                first_dim = next(iter(opts))
                answers[dim] = [opts[first_dim][0]]
            print(f"  {dim} → {answers[dim]}")

        print(f"\nRESUMING with {answers}")
        print("=" * 60)

        async for event in graph.astream_events(
            Command(resume={"action": "submit", "answers": answers}),
            config=config,
            version="v2",
        ):
            name = event.get("name", "")
            kind = event["event"]

            if kind == "on_chain_end" and name == "search_per_group":
                results = event.get("data", {}).get("output", {}).get("results", {})
                for gid, services in results.items():
                    print(f"search_per_group group {gid} → {len(services)} result(s)")
                    for svc in services:
                        dist = f"  {svc.get('distance_km')} km" if svc.get('distance_km') is not None else ""
                        print(f"  service_id={svc.get('service_id')} cat_match={svc.get('category_match')} elig_match={svc.get('eligibility_match')} cat={svc.get('sfsg_category_names')} elig={svc.get('eligibility_all')}{dist}")
                print()

            elif kind == "on_chain_end" and name == "format_results":
                formatted = event.get("data", {}).get("output", {}).get("formatted", {})
                for gid, data in formatted.items():
                    print(f"\nformat_results group {gid}:")
                    print(f"  service_ids: {data.get('service_ids')}")
                    print(f"  rationale: {data.get('rationale')}")
                print()

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk.content, str) and chunk.content:
                    print(chunk.content, end="", flush=True)

    print("\n\nDone.")
    if hasattr(client, "aclose"):
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(run())
