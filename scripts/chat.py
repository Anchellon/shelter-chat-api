"""
Interactive CLI chat — type queries and see the full pipeline output.
Supports multi-turn conversation and HITL interrupts.
Requires: Ollama + MCP server both running.

Run from project root:
    python -m scripts.chat
"""
import asyncio
import json
import sys

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agent.graph import build_graph
from app.core.config import settings


async def stream_query(graph, config, input_):
    """Stream one turn and return interrupt data if any."""
    interrupt_data = None

    async for event in graph.astream_events(input_, config=config, version="v2"):
        name = event.get("name", "")
        kind = event["event"]

        if kind == "on_chain_end" and name == "classify_groups":
            groups = event.get("data", {}).get("output", {}).get("groups", [])
            print(f"\n[{len(group)} group(s) identified]" if (group := groups) else "\n[no groups]")

        elif kind == "on_chain_end" and name == "search_per_group":
            results = event.get("data", {}).get("output", {}).get("results", {})
            for gid, services in results.items():
                print(f"[group {gid}: {len(services)} result(s)]")

        elif kind == "on_chain_end" and name == "format_results":
            formatted = event.get("data", {}).get("output", {}).get("formatted", {})
            for gid, data in formatted.items():
                ids = data.get("service_ids", [])
                print(f"[group {gid}: {len(ids)} service IDs → {ids[:5]}{'...' if len(ids) > 5 else ''}]")

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk.content, str) and chunk.content:
                print(chunk.content, end="", flush=True)

    print()

    state = await graph.aget_state(config)
    for task in state.tasks:
        for intr in getattr(task, "interrupts", []):
            interrupt_data = intr.value if hasattr(intr, "value") else intr

    return interrupt_data


async def handle_interrupt(interrupt_data: dict) -> dict:
    """Prompt the user to fill in HITL gaps interactively."""
    print(f"\n[clarification needed: {interrupt_data.get('group_label', '')}]")
    answers = {}
    for step in interrupt_data.get("steps", []):
        dim = step["dimension"]
        question = step["question"]
        opts = step["options"]

        print(f"\n{question}")
        if isinstance(opts, dict):
            flat = [v for vals in opts.values() for v in vals]
        else:
            flat = opts

        for i, o in enumerate(flat, 1):
            print(f"  {i}. {o}")

        raw = input("Enter number(s) separated by commas, or type your own: ").strip()
        selected = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(flat):
                selected.append(flat[int(part) - 1])
            elif part:
                selected.append(part)

        answers[dim] = selected if len(selected) != 1 else selected[0]

    return answers


async def run():
    print("Connecting to MCP server...")
    client = MultiServerMCPClient({
        "shelter": {"url": settings.mcp_server_url, "transport": "streamable_http"}
    })
    tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    print("Type your query, or 'quit' to exit.\n")

    graph = build_graph(tools, MemorySaver())
    config = {"configurable": {"thread_id": "cli-session-1"}}

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query or query.lower() in ("quit", "exit"):
            print("Bye.")
            break

        interrupt_data = await stream_query(
            graph,
            config,
            {
                "messages": [HumanMessage(content=query)],
                "current_time": "",
                "groups": [],
                "results": {},
                "formatted": {},
            },
        )

        while interrupt_data:
            answers = await handle_interrupt(interrupt_data)
            print(f"\n[resuming with {answers}]\n")
            interrupt_data = await stream_query(
                graph,
                config,
                Command(resume={"action": "submit", "answers": answers}),
            )

    if hasattr(client, "aclose"):
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(run())
