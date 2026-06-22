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
      ├── guardrails            NeMo Guardrails — blocks off-topic or unsafe input
      ├── resolve_intent        LLM → classifies message into one of 8 intents
      │     └── routes to one of:
      ├── classify_groups       LLM → extracts "need groups" (new_search intent)
      ├── refine_groups         LLM → modifies existing groups (refine intent)
      ├── geo_check             Geocodes group locations, drops out-of-SF groups
      ├── intake                LLM → maps groups to categories/eligibilities/geo via MCP tools
      │                         May INTERRUPT here (HITL) if data is missing → POST /api/v1/resume
      ├── search_per_group      MCP search_services call per group
      ├── format_results        LLM → writes rationale per group, returns formatted dict
      ├── converse              Handles follow_up/query intents (mini search agent)
      ├── update_client_context LLM → updates case/group demographics (set_context intent)
      ├── help_node             Returns capabilities list
      ├── acknowledge_node      Sends canned confirmation
      └── clarify_node          Asks clarifying question when intent is ambiguous
```

**Valid intents** (from `resolve_intent`): `new_search`, `refine`, `follow_up`, `query`, `set_context`, `help`, `acknowledge`, `clarify`

Global singletons (set during FastAPI lifespan in `app/main.py`):
- `mcp_client` — `MCPClient` wrapping `MultiServerMCPClient`
- `mcp_tools` — list of LangChain tool objects passed into the graph
- `agent_graph` — compiled `StateGraph`

---

## Key data shapes

### `ClientContext` (`app/agent/state.py`)
```python
age: str | None
housing: str | None
gender: str | None
family_status: str | None
employment: str | None
financial: str | None
health: str | None
ethnicity: str | None
immigration: str | None
language: str | None
other: str | None
```

Helper `effective_context(case_ctx, group_ctx)` merges case-level defaults with per-group overrides.

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
client_context: ClientContext | None  # per-group demographic overrides
```

### `NavigatorState`
```python
messages: list                  # LangGraph message list (add_messages reducer)
groups: list[Group]
results: dict[str, list[dict]]  # group_id → list of service dicts from MCP
formatted: dict[str, dict]      # group_id → {rationale: str, service_ids: [int]}
current_time: str               # sent by frontend, e.g. "Monday 14:30"
intent: str                     # resolved intent type
case_context: ClientContext     # case-level demographic defaults (apply to all groups)
intent_queue: list[str]         # secondary intents queued for next turn
secondary_message: str | None   # message for pending action confirmation
pending_action: str | None      # "clarify", "follow_up", "new_search", "refine", or None
changed_group_ids: list[int]    # groups modified this turn
removed_group_ids: list[int]    # groups dropped this turn
last_query: str | None          # prior org/topic query text
last_query_services: list[dict] # services returned by last query (follow-up source)
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

Both `POST /chat` and `POST /api/v1/resume` return `text/event-stream`. Events emitted:

| type | fields | notes |
|------|--------|-------|
| `text-start` | `id` | start of a new message |
| `text-delta` | `id`, `delta` | streamed text chunk |
| `text-end` | `id` | end of message |
| `tool-start` | `tool`, `status` | human-readable status string |
| `tool-end` | `tool` | |
| `groups_identified` | `groups` | list of Group objects |
| `format_complete` | `formatted`, `groups`, `changed_group_ids`, `removed_group_ids`, `referral_id` | final structured results; `changed_group_ids` lists groups whose search params changed (all on new search, diff on refine); `removed_group_ids` lists groups dropped this turn (refine only); triggers referral creation |
| `intake_request` | `group_id`, `group_label`, `steps` | HITL pause — stream returns early; frontend must POST /api/v1/resume |
| `context_clarify_request` | `group_id`, `group_label`, `steps` | HITL pause for demographic context clarification |
| `clarify_request` | (message) | requests clarification when intent is ambiguous |
| `context_updated` | `case_context`, `groups` | emitted after set_context intent completes |
| `error` | `errorText` | stream aborts |
| `finish` | `finishReason` | always `"stop"` |

Any `*_request` interrupt causes the stream to **return early**. The frontend resumes via `POST /api/v1/resume` with `{conversation_id, action: "submit"|"cancel", answers: {...}}`.

---

## MCP tools (from shelter-mcp-server)

| tool | purpose |
|------|---------|
| `list_categories` | returns list of category strings (e.g. `"sfsg-shelter"`) |
| `list_eligibilities` | returns dict of eligibility groups keyed by dimension |
| `geocode_location` | `{location_text}` → `{lat, lng}` via Nominatim (OpenStreetMap) |
| `search_services` | `{query, categories?, eligibilities?, lat?, lng?, radius_ft?, when?}` → list of service dicts (semantic vector search) |
| `search_by_name` | `{name}` → list of service dicts (substring search, used by `converse` query path) |
| `get_service_details` | single service by id |
| `get_service_details_batch` | batch fetch by id list |

MCP results come back as `[{"type": "text", "text": "<json>"}]` — always unwrap via `_unwrap_tool_result()` (defined in both `intake.py` and `mcp_client.py`) before use.

---

## LLM configuration

Three independently configurable LLM roles in `.env` / `config.py`:

| role | config keys | used in |
|------|-------------|---------|
| classifier | `classifier_provider`, `classifier_model` | `classify_groups`, `refine_groups`, `resolve_intent` nodes |
| intake | `intake_provider`, `intake_model` | `intake`, `update_client_context` nodes |
| formatter | `formatter_provider`, `formatter_model` | `format_results`, `converse` nodes |

Provider options: `"anthropic"` (default), `"openai"`, `"ollama"`. Default model: `claude-haiku-4-5-20251001`. Factory in `app/agent/llm.py`.

---

## Grounding

All factual content (org names, addresses, hours, eligibilities) flows from Postgres through MCP tools — there is **no general-knowledge fallback**. `search_per_group` and the converse query mini-agent (`_handle_query` in `app/agent/nodes/converse.py`) only render what their tools return; the system prompts explicitly forbid inventing data.

The LLM is still the rendering layer, so the constraint is **soft, not hard**:

- **Follow-ups** (`_handle_follow_up`) answer from `results` and `last_query_services` already in state — no fresh tool calls. Cited fields (addresses, service IDs, names) come straight out of saved dicts and are essentially safe.
- **Query path** (`_handle_query`) tool-calls and synthesizes; outputs the LLM's prose grounded in tool results plus captures the raw services into `last_query` / `last_query_services` for the next turn.
- **Derived/comparative statements** ("closer to BART", "good fit for your client") are LLM reasoning on top of grounded data — first place drift would show up if it ever does.

If hallucinations become an issue, options are: structured output instead of prose, an evaluator pass that verifies claims against tool results, or surfacing the source field name inline in the rendered answer.

---

## Database

PostgreSQL via `psycopg` (async). Migrations managed with Flyway (`migrations/V*.sql`).

| table | purpose |
|-------|---------|
| `checkpoints` + related | LangGraph conversation state (managed by `AsyncPostgresSaver`) |
| `conversation_summaries` | `(thread_id, user_id, title)` — index of past conversations |
| `referrals` | `(id UUID, user_id, thread_id, title, saved bool, groups JSONB, changed_group_ids JSONB, removed_group_ids JSONB)` — search results saved per turn, plus refine-diff metadata |
| `saved_services` | `(id UUID, user_id, service_id int)` — individual services bookmarked by a navigator; unique on `(user_id, service_id)` |

**`referrals.groups`** is a JSONB array of merged Group + formatted objects:
```json
[{"group_id": 1, "what": "...", ..., "rationale": "...", "service_ids": [123]}]
```

**`referrals.changed_group_ids`** / **`referrals.removed_group_ids`** are JSONB arrays of `group_id` integers, used by the frontend to render the "Updated"/"Removed" affordances per turn on reload. Both default to `[]`.

DB helpers live in `app/core/db.py`: `save_conversation_summary()`, `create_referral()`. Direct psycopg queries elsewhere in API routes use `psycopg.AsyncConnection.connect(settings.database_url)`.

---

## Auth

`app/core/auth.py` — `require_user` FastAPI dependency. Validates Auth0 JWT (RS256), checks `navigator-api/roles` claim is non-empty, returns `user_id` (Auth0 `sub`). If `auth0_domain` is not set in env, auth is **disabled** and returns `"dev"` — used in local development.

---

## API routes

| method | path | purpose |
|--------|------|---------|
| `POST` | `/api/v1/chat` | Start or continue a conversation (SSE) |
| `POST` | `/api/v1/resume` | Resume after any HITL interrupt (SSE) |
| `GET` | `/api/v1/conversations` | List conversations for current user |
| `GET` | `/api/v1/conversations/{id}` | Full conversation state + referrals |
| `POST` | `/api/v1/services/batch` | Fetch service details by id list (via MCP) |
| `POST` | `/api/v1/referrals` | Create a referral manually |
| `PATCH` | `/api/v1/referrals/{id}` | Update referral title and/or saved flag |
| `GET` | `/api/v1/referrals` | List saved referrals for current user |
| `GET` | `/api/v1/referrals/{id}` | Get a single referral |
| `DELETE` | `/api/v1/referrals/{id}` | Delete a referral |
| `POST` | `/api/v1/saved-services` | Save (bookmark) a service by service_id |
| `DELETE` | `/api/v1/saved-services/{service_id}` | Remove a bookmarked service |
| `GET` | `/api/v1/saved-services` | List bookmarked services for current user |
| `GET` | `/health` | Liveness check — returns `{status, mcp_connected, agent_ready}` |

---

## Conventions

- **Node factories**: `build_*_node(tools_by_name)` pattern used for nodes that need MCP tools — they close over the tools dict and return the async node function. See `intake.py`, `search_per_group.py`, `format_results.py`.
- **New migrations**: add `migrations/V{N+1}__description.sql`. Never edit existing migration files.
- **No ORM**: raw `psycopg` async queries throughout. No SQLAlchemy.
- **Agent graph globals**: `agent_graph`, `mcp_client`, `mcp_tools` are module-level globals in `app/main.py`, imported inside functions to avoid circular imports.
- **Guardrails**: NeMo Guardrails config lives in `app/guardrails/config/`. Engine is Claude Haiku (Anthropic). On timeout or error it fails open (allows the message through).
- **Streaming**: all chat endpoints return `StreamingResponse` with `text/event-stream`. Never return JSON directly from chat routes.
- **Intent routing**: `resolve_intent` is the central dispatcher. Any new behaviour should map to one of the 8 existing intents or a new intent added to `resolve_intent` — don't branch on message content elsewhere in the graph.
- **Dual context**: `case_context` is case-level defaults; `group.client_context` overrides per person. Always merge via `effective_context()` before building eligibility lists.

---

## System context

This repo is one of four in the Navigator system:

| repo | purpose |
|------|---------|
| `shelter-chat-api` (this repo) | FastAPI + LangGraph agent backend |
| `shelter-search` | React 19 + TypeScript SPA (the navigator UI) |
| `shelter-mcp-server` | FastMCP server — semantic search over SF shelter/services data (pgvector + Bedrock Titan embeddings) |
| `shelter-infra` | AWS CDK IaC — ECS Fargate, RDS PostgreSQL 16, CloudFront, Langfuse/Grafana/Sentry observability |

**Deployment** (from `shelter-infra`):
- Only a single **Navigator-Staging** environment is deployed; it carries real navigator traffic (treated as prod)
- Agent API: CloudFront → ALB → ECS Fargate (chat-api, port 3000)
- MCP server: internal DNS `mcp.navigator-staging.internal:8001` — not publicly accessible
- Frontend: S3 + CloudFront
- Database: RDS PostgreSQL 16 (`shelter` db), private subnet only
- Embeddings: Amazon Titan Embed Text v2 via AWS Bedrock

**Observability** (infrastructure-level, OTEL collector sidecar on each Fargate task):
- Traces → Langfuse (`us.cloud.langfuse.com`)
- Metrics → Grafana Cloud (instance 1642287)
- Errors → Sentry (separate DSNs for chat-api, mcp-server, frontend)
