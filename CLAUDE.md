# shelter-chat-api — Claude Reference

## What this is

A FastAPI backend for a social services navigator chat. Social workers ("navigators") describe client needs in natural language; an agentic pipeline searches shelter/food/services data and returns structured results. Runs on LangGraph + MCP + PostgreSQL.

---

## Architecture

```
POST /api/v1/chat
      │
      ▼
  agent_graph (LangGraph, compiled at startup)
      │
      ├── guardrails      NeMo Guardrails — blocks off-topic or unsafe input
      ├── classify_groups LLM → extracts "need groups" from user message
      ├── intake          LLM → maps groups to categories/eligibilities/geo via MCP tools
      │                   May INTERRUPT here (HITL) if data is missing → POST /chat/resume
      ├── search_per_group MCP search_services call per group
      └── format_results  LLM → writes rationale per group, returns formatted dict
```

Global singletons (set during FastAPI lifespan in `app/main.py`):
- `mcp_client` — `MCPClient` wrapping `MultiServerMCPClient`
- `mcp_tools` — list of LangChain tool objects passed into the graph
- `agent_graph` — compiled `StateGraph`

---

## Key data shapes

### `Group` (`app/agent/state.py`)
```python
group_id: int
what: str           # "food and shelter"
who: str | None     # "LGBTQ teens"
where: str          # defaults to "San Francisco"
when: str | None    # "Saturday morning"
open_now: bool
# populated by intake:
categories: list[str]   # e.g. ["sfsg-shelter"]
eligibilities: list     # e.g. ["Adults", "Anyone in Need"]
lat: float | None
lng: float | None
```

### `NavigatorState`
```python
messages: list          # LangGraph message list (add_messages reducer)
groups: list[Group]
results: dict[str, list[dict]]  # group_id → list of service dicts from MCP
formatted: dict[str, dict]      # group_id → {rationale: str, service_ids: [int]}
current_time: str               # sent by frontend, e.g. "Monday 14:30"
```

### `formatted` dict (output of `format_results_node`)
```python
{
  "1": {"rationale": "...", "service_ids": [123, 456]},
  "2": {"rationale": "...", "service_ids": [789]},
}
```

---

## SSE event stream

Both `POST /chat` and `POST /chat/resume` return `text/event-stream`. Events emitted:

| type | fields | notes |
|------|--------|-------|
| `text-start` | `id` | start of a new message |
| `text-delta` | `id`, `delta` | streamed text chunk |
| `text-end` | `id` | end of message |
| `tool-start` | `tool`, `status` | human-readable status string |
| `tool-end` | `tool` | |
| `groups_identified` | `groups` | list of Group objects |
| `format_complete` | `formatted`, `groups`, `referral_id` | final structured results; triggers referral creation |
| `intake_request` | `group_id`, `group_label`, `steps` | HITL pause; frontend must POST /chat/resume |
| `error` | `errorText` | stream aborts |
| `finish` | `finishReason` | always `"stop"` |

`intake_request` causes the stream to **return early**. The frontend resumes via `POST /api/v1/chat/resume` with `{conversation_id, action: "submit"|"cancel", answers: {...}}`.

---

## MCP tools (from shelter-search MCP server)

| tool | purpose |
|------|---------|
| `list_categories` | returns list of category strings (e.g. `"sfsg-shelter"`) |
| `list_eligibilities` | returns dict of eligibility groups |
| `geocode_location` | `{location_text}` → `{lat, lng}` |
| `search_services` | `{query, categories?, eligibilities?, lat?, lng?, when?}` → list of service dicts |
| `get_service_details` | single service by id |
| `get_service_details_batch` | batch fetch by id list |

MCP results come back as `[{"type": "text", "text": "<json>"}]` — always unwrap via `_unwrap_tool_result()` (defined in both `intake.py` and `mcp_client.py`) before use.

---

## LLM configuration

Three independently configurable LLM roles in `.env` / `config.py`:

| role | config keys | used in |
|------|-------------|---------|
| classifier | `classifier_provider`, `classifier_model` | `classify_groups` node |
| intake | `intake_provider`, `intake_model` | `intake` node (category/eligibility mapping) |
| formatter | `formatter_provider`, `formatter_model` | `format_results` node |

Provider options: `"ollama"` (default), `"openai"`, `"anthropic"`. Factory in `app/agent/llm.py`.

---

## Database

PostgreSQL via `psycopg` (async). Migrations managed with Flyway (`migrations/V*.sql`).

| table | purpose |
|-------|---------|
| `checkpoints` + related | LangGraph conversation state (managed by `AsyncPostgresSaver`) |
| `conversation_summaries` | `(thread_id, user_id, title)` — index of past conversations |
| `referrals` | `(id UUID, user_id, thread_id, title, saved bool, groups JSONB)` — search results saved per turn |

**`referrals.groups`** is a JSONB array of merged Group + formatted objects:
```json
[{"group_id": 1, "what": "...", ..., "rationale": "...", "service_ids": [123]}]
```

DB helpers live in `app/core/db.py`: `save_conversation_summary()`, `create_referral()`. Direct psycopg queries elsewhere in API routes use `psycopg.AsyncConnection.connect(settings.database_url)`.

---

## Auth

`app/core/auth.py` — `require_user` FastAPI dependency. Validates Auth0 JWT (RS256), checks `navigator-api/roles` claim is non-empty, returns `user_id` (Auth0 `sub`). If `auth0_domain` is not set in env, auth is **disabled** and returns `"dev"` — used in local development.

---

## API routes

| method | path | purpose |
|--------|------|---------|
| `POST` | `/api/v1/chat` | Start or continue a conversation (SSE) |
| `POST` | `/api/v1/chat/resume` | Resume after HITL intake interrupt (SSE) |
| `GET` | `/api/v1/conversations` | List conversations for current user |
| `GET` | `/api/v1/conversations/{id}` | Full conversation state + referrals |
| `POST` | `/api/v1/services/batch` | Fetch service details by id list (via MCP) |
| `POST` | `/api/v1/referrals` | Create a referral manually |
| `PATCH` | `/api/v1/referrals/{id}/save` | Star/save a referral |
| `GET` | `/api/v1/referrals` | List saved referrals for current user |
| `GET` | `/api/v1/referrals/{id}` | Get a single referral |
| `DELETE` | `/api/v1/referrals/{id}` | Delete a referral |
| `GET` | `/health` | Liveness check |

---

## Conventions

- **Node factories**: `build_*_node(tools_by_name)` pattern used for nodes that need MCP tools — they close over the tools dict and return the async node function. See `intake.py`, `search_per_group.py`, `format_results.py`.
- **New migrations**: add `migrations/V{N+1}__description.sql`. Never edit existing migration files.
- **No ORM**: raw `psycopg` async queries throughout. No SQLAlchemy.
- **Agent graph globals**: `agent_graph`, `mcp_client`, `mcp_tools` are module-level globals in `app/main.py`, imported inside functions to avoid circular imports.
- **Guardrails**: NeMo Guardrails config lives in `app/guardrails/config/`. On timeout or error it fails open (allows the message through).
- **Streaming**: all chat endpoints return `StreamingResponse` with `text/event-stream`. Never return JSON directly from chat routes.
