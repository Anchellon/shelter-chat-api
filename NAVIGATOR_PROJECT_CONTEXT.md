# Navigator — Project Context for Cover Letter Drafting

> **Purpose of this document.** Self-contained technical briefing covering all five repositories that make up the Navigator platform. Designed to be loaded into a fresh Claude project as the sole context needed to write technical cover letters about this work. Every claim here is grounded in the actual code as of 2026-06-22.

---

## 1. One-paragraph executive summary

**Navigator** is a full-stack, AI-powered social-services navigation platform that lets case workers ("navigators") describe a client's needs in natural language and receive structured, grounded referrals to San Francisco shelter, food, and human-services providers. The system is a microservice deployment on AWS comprising a streaming FastAPI agent backend, a FastMCP semantic-search server backed by PostgreSQL + pgvector, an SQL-driven ingestion pipeline that produces 1024-dim Bedrock Titan embeddings, a React 19 single-page client with custom SSE streaming, and an AWS CDK (Python) infrastructure-as-code repo that provisions everything via federated OIDC GitHub Actions. The platform is live in a `Navigator-Staging` AWS account carrying real navigator traffic.

---

## 2. The problem and the user

- **User**: SF social-services navigators / case workers — non-engineers who need to find appropriate referrals for clients (shelter, food, jobs, health, etc.) under time pressure.
- **Why an LLM**: Source data lives in a normalized OpenReferral-style PostgreSQL schema with eligibility tags, categories, schedules, and free-text descriptions. Mapping a sentence like *"need shelter for an LGBTQ teen on Saturday morning"* to the right category + eligibility + geo + open-hours filter is exactly what an LLM-routed semantic search excels at.
- **Why grounded**: All factual content (org names, addresses, hours, eligibilities) flows from Postgres through MCP tools. The agent has **no general-knowledge fallback** — system prompts explicitly forbid invented data. The LLM is a rendering/routing layer over a structured tool surface.
- **Why streaming + HITL**: Searches take a few seconds; partial UI updates matter. When the agent can't unambiguously resolve a category or eligibility, it interrupts (LangGraph `interrupt()`) and asks the navigator a clarifying multi-step question — the frontend renders this as an intake form, then resumes the graph.

---

## 3. System architecture (end-to-end)

```
Navigator (React SPA, Vite)
   │  HTTPS (Auth0 JWT) + SSE stream
   ▼
CloudFront (forced HTTP/1.1 to preserve SSE)
   ▼
Application Load Balancer (public, :80)
   ▼
ECS Fargate ── shelter-chat-api (FastAPI + LangGraph agent)
   │   - guardrails → resolve_intent → (classify | refine | converse | …)
   │   - HITL interrupt via LangGraph
   │   - AsyncPostgresSaver checkpointing
   │   ↑ MCP over HTTP at mcp.navigator-staging.internal:8001
   ▼
ECS Fargate ── shelter-mcp-server (FastMCP)
   │   - 7 tools: search_services, search_by_name,
   │     get_service_details(_batch), geocode_location,
   │     list_categories, list_eligibilities
   │   - Cosine-distance pgvector search + tag/geo/hours reranking
   ▼
RDS PostgreSQL 16 + pgvector (private isolated subnet)
   ▲
EventBridge nightly cron @ 10:00 UTC
   │
   ▼
ECS Fargate (transient) ── ingestion-pipeline
   - SQL-driven denormalization (12-CTE query, 637 LOC)
   - AWS Bedrock Titan v2 embeddings (1024-dim)
   - SSM Parameter Store last_run_at for incremental loads
```

All inter-service discovery uses **AWS Cloud Map** private DNS (`*.navigator-staging.internal`) — no hardcoded IPs, services scale horizontally without config changes. Observability sidecar is **ADOT (AWS Distro for OpenTelemetry)** fanning metrics/traces to **Langfuse + Grafana Cloud + Sentry** (Langfuse migration is recent — replaced an older LangSmith setup).

---

## 4. Per-repo deep dive

### 4.1 `shelter-chat-api` — agent backend (~4,022 LOC, 37 Python files)

**Stack**: Python 3.12, FastAPI 0.115, LangGraph 0.2 with `AsyncPostgresSaver` checkpointer, LangChain MCP adapters, NeMo Guardrails 0.10, raw `psycopg` async (no ORM), Auth0 RS256 JWT via `python-jose`, Flyway migrations (8 versioned files), Langfuse + OpenTelemetry.

**Agent graph** (13 nodes, compiled at FastAPI startup):
- `guardrails` — Claude-Haiku-gated NeMo safety/scope/advice filter; **fails open** on 30s timeout to protect UX.
- `resolve_intent` — classifies into 8 intents: `new_search`, `refine`, `follow_up`, `query`, `set_context`, `help`, `acknowledge`, `clarify`. Ordinal references like *"the second group"* are routed to `refine`, not a new search.
- `classify_groups` / `refine_groups` — extract structured `Group` records (what / who / where / when) plus a `client_context` block of structured demographics (age, housing, gender, family_status, employment, financial, health, ethnicity, immigration, language). Refine preserves categories/eligibilities/lat/lng on unchanged groups so we don't re-LLM-map and re-geocode; emits `changed_group_ids` + `removed_group_ids` for frontend diff affordances.
- `geo_check` — geocodes per-group `where`, drops groups outside the SF bounding box (37.63–37.84 lat, –122.52 to –122.35 lng).
- `intake` — LLM-maps `what` → MCP categories, `who` → MCP eligibilities. If gaps remain, calls `langgraph.types.interrupt({"group_id", "group_label", "steps": […]})`; stream pauses, frontend gets an `intake_request` SSE event, navigator answers, `POST /api/v1/resume` calls `graph.invoke(Command(resume=…))`.
- `search_per_group` — one MCP `search_services` call per group, then `get_service_details_batch` enriches the top 5.
- `format_results` — generates 1-2-sentence rationales; emits `formatted: dict[group_id → {rationale, service_ids}]`.
- `converse` — out-of-graph queries; capped at 3 tool iterations; follow-ups answer from cached `last_query_services` without re-fetching.
- `update_client_context`, `help_node`, `acknowledge_node`, `clarify_node` — meta-responses; `update_client_context` can also HITL-interrupt when scope is ambiguous (emits `context_clarify_request`, resumes via `POST /api/v1/resume`), and can chain another intent via `intent_queue` when context changes affect eligibility.

**Engineering choices worth surfacing**:
- **AsyncPostgresSaver** for conversation state (multi-process resume by `thread_id`, no in-memory drift).
- Custom **`with_heartbeat()` SSE wrapper** emitting a `_heartbeat` sentinel every 15s to defeat CloudFront / nginx idle-timeout without cancelling in-flight LLM requests.
- Three independently configurable LLM roles (`classifier`, `intake`, `formatter`) so each task can swap between Ollama / OpenAI / Anthropic without code changes.
- **Synthetic `AIMessage` referral marker** (`id=f"referral_{uuid}"`) is `aupdate_state`-injected into the checkpoint so `GET /conversations/{id}` can reconstruct exact turn order without positional heuristics.
- Auth0 dev-mode bypass — if `auth0_domain` is unset, `require_user` returns `"dev"`. Lets local development work without an Auth0 tenant.

**API surface**: `POST /chat`, `POST /api/v1/resume` (HITL resume), `GET /conversations[/:id]`, `POST /services/batch`, full referral CRUD (`POST/PATCH/GET/DELETE /referrals`), saved-services bookmarks (`POST/DELETE/GET /saved-services`), `/health`.

**Deployment**: Dockerfile (python:3.12-slim + g++ for psycopg), GitHub Actions OIDC into AWS, ECR push, `ecs:UpdateService` rollout.

---

### 4.2 `shelter-mcp-server` — semantic search MCP (~661 LOC, 7 Python modules)

**Stack**: Python 3.12, FastMCP 2.0 (HTTP transport on :8001), asyncpg (pool 2–10 conns), `langchain-aws` BedrockEmbeddings (Titan v2, 1024-dim), Uvicorn, httpx for Nominatim. **No ORM** — hand-tuned SQL with window functions and lateral patterns.

**Search ranking (`search_services`)**:
1. Embed the query via Bedrock → cosine distance against `service_snapshots.embedding` (filter `similarity < 0.8`).
2. Optional **schedule filter**: parse `"Monday 14:00"` → `(day, minutes_since_midnight)`; a service passes if it has no schedule, or any entry where `open_mins ≤ q ≤ close_mins`.
3. Optional **geo filter**: haversine vs `radius_ft` (default 2500 ft ≈ 0.76 km).
4. **Tag nudges, not hard filters**: a category overlap subtracts 0.10 from the score, eligibility overlap subtracts 0.05. Semantic matches outside tag space still surface; tag matches just outrank them.
5. Sort by adjusted similarity, tiebreak by distance, dedupe via `DISTINCT ON (service_id)`, cap at 50.

**`search_by_name` round-robin ranking**: uses `ROW_NUMBER() OVER (PARTITION BY resource_id)` plus `DENSE_RANK()` over match quality (exact → prefix → substring) to **interleave** multi-location orgs — first location of every YMCA branch before second locations — so a single org can't monopolize results.

**Unified `_build_detail_sql` helper**: one parameterized SQL template with service-level → resource-level fallback for address / phone / notes is shared by `get_service_details`, `get_service_details_batch`, and `search_by_name`. Guarantees consistent schemas across entry points; the LLM downstream always sees the same shape.

**Embedding text** is intentionally **excluded** from detail responses — the prose blob (30–50% of payload size) is for vector search only; every field it contained is now exposed as a structured column to keep the LLM context window tight.

`geocode_location` uses **OSM Nominatim** with an SF viewbox preference and global fallback; 5s timeout, no caching.

---

### 4.3 `ingestion-pipeline` — denormalize + embed (~850 LOC Python + ~786 LOC SQL)

This is a **semantic indexing pipeline**, not an upstream data scraper — it consumes pre-loaded OpenReferral-ish PostgreSQL tables and produces the `service_snapshots` table that `shelter-mcp-server` queries.

**SQL-driven (the heavy lifting is in `sql/service_snapshot.sql`, 637 lines, 12 CTEs)**:
- Joins services × resources × addresses × phones × schedules × categories × eligibilities × notes.
- Fans out to **one row per service-address combo** for multi-location services (enables per-location geo filtering without array unwrapping).
- **Eligibility remapping** (64 explicit `CASE` branches) — e.g. `"Adolescents" → "Youth (below 21 years old)"`.
- **Eligibility bucketing** into 11 dimensions (age, education, employment, ethnicity, family_status, financial, gender, health, immigration, housing, other) stored as 11 `text[]` columns plus an `eligibility_all` rollup — backs faceted search without secondary queries.
- **Schedule normalization** — integer-encoded hours (`900`, `1730`) become `"9:00 AM – 5:30 PM"` prose + a JSONB array `[{day, open_mins, close_mins}]` for "open now" filtering.
- All denormalization is deterministic SQL — **no LLM enrichment**, no classification, no summarization. The pipeline is fully reproducible.

**Embedding**:
- **AWS Bedrock Titan v2** (1024-dim), batch_size 25, 10 req/s steady state to stay under quota.
- **Boto3 directly, not LangChain's `BedrockEmbeddings`** — the LangChain wrapper has its own 4-attempt retry that compounded with the pipeline's retry loop into a double-retry problem; calling Bedrock directly gives a single predictable exponential-backoff-with-jitter loop.
- Embedding dimension is verified against config at first-batch runtime (raises `ValueError` on mismatch).

**Incremental loads**: `last_run_at` lives in **AWS SSM Parameter Store** (no separate state table, survives container restarts). Incremental run = `WHERE services.updated_at > last_run_at`, plus pulls soft-deleted (`status != 1`) IDs to evict stale snapshots. Full and incremental writes run inside **one transaction** (TRUNCATE + INSERT or DELETE + INSERT) so readers never see partial state — zero-downtime swap.

**Output schema** (`service_snapshots`): HNSW index on `embedding` with `vector_cosine_ops`, GIN indexes on category/eligibility text arrays, B-tree on IDs and lat/lng.

**Deployment**: Docker → ECR → EventBridge cron @ 10:00 UTC → transient Fargate task. GitHub Actions builds + pushes via OIDC on push to `main`.

---

### 4.4 `shelter-search` — React 19 SPA (~4,758 LOC, 50 TS/TSX files)

**Stack**: Vite + React 19 + TypeScript 5.9, Tailwind v3 (custom palette — `#276ce5` primary, deliberately sparse), Redux Toolkit (4 slices: chat / ui / conversations / user), React Router v7, Auth0 React SDK, Zod for runtime validation of streamed payloads, react-markdown for AI message rendering. **No TanStack Query / SWR**, **no Vercel AI SDK** — fetch + a hand-written SSE parser.

**Custom SSE consumer** (`src/services/api.ts`):
- Spec-correct: events split on `\n\n`, multi-line `data:` concatenated with `\n`.
- `ReadableStream.getReader()` + `TextDecoder` driven by an `async function* parseSSE()` — events yield as they arrive.
- Handles every event type emitted by the backend: `text-start` / `text-delta` / `text-end`, `tool-start` / `tool-end`, `groups_identified`, `format_complete`, `intake_request`, `context_clarify_request`, `context_updated`, `clarify_request`, `finish`, `error`.

**Stream-buffering pattern**: incoming text deltas accumulate in `pendingText` Redux state and **are not rendered as a finalized message** until `format_complete`, `clarify_request`, or `finish` arrives. Eliminates flicker during multi-step agent reasoning while still showing live typing.

**Other patterns worth highlighting**:
- **Compound group keys** `${referralId}_${groupId}` for paginated result caches — same `group_id` can recur across referrals (multi-turn), so the cache key has to disambiguate.
- **Lazy service-detail batch fetch** — service cards page through results client-side; `fetchServicesBatch()` only requests IDs not already cached.
- **Virtual step expansion of the intake form** — backend sends one `intake_request` event with nested options; the frontend expands it client-side into a multi-step wizard with keyboard nav, "Something else" fallback, and a progress bar.
- **Auth0 token wiring via `setTokenGetter()`** — `AppProviders` hands the Auth0 `getAccessTokenSilently` function to the API service module so every fetch attaches a Bearer token with no prop drilling.
- **Referral idempotency** — referral-card messages are keyed by `referralId` and replaced (not appended) on re-emission, so a re-processed referral never duplicates.

**Surfaces**: landing (search + 4 prompt chips), chat (two-pane: ChatPane + ResultsPane), referrals (saved collections with rename / duplicate / delete), recents, saved-services. Responsive (Tailwind `md:` breakpoints, single-pane toggling on mobile). ARIA labels on chat log + conversation, `role="alert"` for errors.

`mockups/` holds hand-coded HTML/CSS design-phase artifacts (`landing.html`, `chat.html`, `collections.html`) used for stakeholder review before the React build.

---

### 4.5 `shelter-infra` — AWS CDK Python IaC (8 stack files + 4 shell scripts)

**Stack**: AWS CDK in Python. Synthesizes **6 CloudFormation stacks** (3 per environment) covering Network, Database, MCP, Agent, Ingestion, Frontend. Account `746669221991` / region `us-east-1`. CDK feature flags pinned in `cdk.json`.

**Deployed resources**:
- **VPC** multi-AZ with public + private-isolated subnets. **No NAT gateway** — Fargate tasks get public IPs directly (saves ~$32/mo/env, trades a small surface increase).
- **RDS PostgreSQL 16** with pgvector, encrypted at rest, 7-day automated backups, private-isolated, deletion-protected in prod, t3.micro in staging.
- **ECS Fargate** for chat-api (1 vCPU / 2 GB), MCP (512 / 1 GB), ingestion (transient 1 vCPU / 2 GB on EventBridge schedule). Autoscale 1–4 tasks at 70% CPU/memory.
- **ALB → Fargate** on `:80` → container `:3000`, health-checked via `/health`.
- **CloudFront** for API (forced **HTTP/1.1** — HTTP/2 chunked SSE incorrectly) and for the S3-hosted frontend bundle.
- **AWS Cloud Map** private DNS namespace `navigator-{env}.internal` for service discovery — chat-api resolves `mcp.navigator-staging.internal:8001`.
- **Secrets Manager** for RDS creds (auto-generated), Anthropic API key, Auth0 config, Grafana / Langfuse / Sentry tokens.
- **SSM Parameter Store** for the ingestion pipeline's `last_run_at` and operational metadata (cluster / service names).
- **EventBridge** nightly cron @ 10:00 UTC triggers the ingestion Fargate task; CloudWatch metric filter on `ERROR|CRITICAL|FATAL` → SNS → email alert.
- **AWS Bedrock** — `amazon.titan-embed-text-v2:0` invoked on demand by both MCP and ingestion. IAM permission scoped to the specific model ARN.
- **GitHub Actions OIDC** federation — no static AWS keys. Per-service per-environment roles: `navigator-{mcp|chatapi|ingestion|frontend}-github-role`, each scoped to `repo:Anchellon/{repo-name}:*` with least-privilege ECR push + `ecs:UpdateService` (or S3 / CloudFront invalidation for the frontend).

**Pragmatic patterns worth calling out**:
- **Single staging environment treated as production** — `Navigator-Staging` carries real navigator traffic; the same stack patterns apply to `Navigator-Prod` (defined but not deployed). The infra design makes the cutover a config-only change.
- **CloudFront HTTP/1.1 for SSE** — a non-obvious gotcha; HTTP/2 multiplexing on CloudFront breaks server-sent event streams.
- **Pre-baked Ollama embedding model** in the MCP image (Dockerfile downloads `nomic-embed-text` at build time) — eliminates cold-start latency on the fallback path.
- **Observability sidecar pattern** — ADOT collector runs as a `non-essential` container; if it crashes the app keeps serving traffic. Metrics fan out to Grafana Cloud + Langfuse + Sentry.
- **Database restore script** (`scripts/restore_db.sh`) — spins up a temporary `postgres:16-alpine` Fargate task with presigned S3 URL + temporary IAM role + temporary SG to stream a SQL dump straight into the DB, then `trap cleanup EXIT` tears everything down. No permanent ops connectivity needed.

---

## 5. Tech-stack matrix (at a glance)

| Layer | Tech |
|------|------|
| Frontend | React 19, TypeScript 5.9, Vite, Tailwind v3, Redux Toolkit, React Router v7, Auth0 SDK, Zod |
| Streaming protocol | Server-Sent Events with custom typed event vocabulary (13 event types) |
| API | FastAPI 0.115, Uvicorn, Pydantic Settings, Auth0 RS256 JWT |
| Agent runtime | LangGraph 0.2 with AsyncPostgresSaver, LangChain MCP adapters, NeMo Guardrails 0.10 |
| LLM providers | Anthropic (Claude Haiku 4.5 default for classifier/intake/formatter), Ollama (Qwen 2.5-7b local), OpenAI (factory-pluggable) |
| MCP server | FastMCP 2.0 over HTTP, 7 tools |
| Embeddings | AWS Bedrock Titan Text Embeddings v2 (1024-dim) primary; nomic-embed-text fallback |
| Search | PostgreSQL 16 + pgvector HNSW (cosine) + GIN on text[] tag columns + B-tree geo |
| DB driver | asyncpg (MCP), psycopg 3 async (chat-api & ingestion) — no ORM anywhere |
| Migrations | Flyway (8 versioned SQL files) |
| Ingestion | SQL-driven (12-CTE denorm, 637 LOC) + Bedrock embeddings + SSM-tracked incremental |
| Cloud | AWS only — VPC, RDS, ECS Fargate, ALB, CloudFront, S3, Cloud Map, EventBridge, Secrets Manager, SSM, Bedrock, IAM, CloudWatch, SNS |
| IaC | AWS CDK (Python), 6 stacks per env |
| CI/CD | GitHub Actions with AWS OIDC federation (no static keys), per-service per-env IAM roles |
| Observability | OpenTelemetry / ADOT sidecar → Langfuse + Grafana Cloud + Sentry |
| Auth | Auth0 OIDC (RS256 JWKS), navigator-role claim required, dev-mode bypass |

---

## 6. Engineering decisions worth highlighting in cover letters

These are the items most worth pulling into a cover letter because they show judgment, not just tool-use:

1. **Grounding by construction.** No general-knowledge fallback — every factual field rendered to a navigator comes back from a tool call against Postgres. Soft-enforced via prompts; verifiable by inspection. Mentioned explicitly in the chat-api's CLAUDE.md as the trust boundary.
2. **HITL inside the graph, not outside.** LangGraph `interrupt()` lets the agent pause, the frontend collect structured answers via a wizard, and the graph resume from the exact node — no external state machine.
3. **Refine = diff, not re-search.** When a navigator iterates on a turn, `refine_groups` preserves resolved categories/eligibilities/geo on unchanged groups and emits `changed_group_ids` + `removed_group_ids` so the UI shows "Updated" / "Removed" affordances instead of redrawing everything.
4. **Custom SSE infrastructure both ends.** Server-side `with_heartbeat()` defeats CloudFront idle-timeout without cancelling the LLM; client-side hand-written `parseSSE` async generator + Redux `pendingText` buffering for jank-free streaming. CloudFront forced to HTTP/1.1 because HTTP/2 mis-chunks SSE.
5. **Tag matches as score nudges, not filters** in `search_services` (–0.10 for category overlap, –0.05 for eligibility). Keeps the semantic search permissive while still respecting structured intent.
6. **Round-robin org interleaving** in `search_by_name` via `ROW_NUMBER() OVER (PARTITION BY resource_id)` so multi-location orgs don't monopolize a name lookup.
7. **SQL-first ingestion** — 12-CTE denormalization plus 64 hardcoded eligibility remappings live in version-controlled SQL, not a Python pipeline that re-implements the same logic in objects. Deterministic, debuggable, fast.
8. **Boto3-direct embeddings to fix a double-retry bug** — LangChain's `BedrockEmbeddings` wrapper retries internally; combined with the pipeline's own retry loop it burned quota. Dropping to boto3 collapsed it to one predictable backoff loop.
9. **Single-transaction snapshot swap** in ingestion — readers never see partial data because TRUNCATE+INSERT (full) or DELETE+INSERT (incremental) both commit atomically.
10. **Cloud Map service discovery + OIDC IAM** — no hardcoded service URLs, no static AWS credentials anywhere in the codebases.
11. **AsyncPostgresSaver for LangGraph state** — multi-process resume, no in-memory drift, durable conversation history that survives container rotations.
12. **Three independently configurable LLM roles** so each task (classifier / intake / formatter) can swap providers without code changes — useful for cost/latency experimentation.
13. **Three observability sinks** (Langfuse for LLM traces, Grafana Cloud for infra metrics, Sentry for exceptions) fan-out via one ADOT collector as a non-essential sidecar — degrades open if observability breaks.

---

## 7. Rough scale indicators

- **5 repos**, all owned end-to-end by one engineer.
- **~10,000 LOC** of application code (Python + TypeScript) plus **~800 LOC** of SQL plus **~1,200 files** of CDK/CloudFormation.
- **6 CloudFormation stacks** synthesized from ~9 Python CDK files.
- **8 Flyway migrations** on the chat-api side, **2 DDL files** on the ingestion side.
- **13 LangGraph nodes** in the agent, **7 MCP tools** in the search server, **13 SSE event types** in the streaming contract.
- **1024-dim** embeddings indexed via HNSW with cosine distance, capped at 50 results per query.
- Live in **AWS us-east-1**, single environment, real navigator traffic.

---

## 8. What this project is *not*

To save the cover-letter Claude from over-claiming:

- Not multi-region, not multi-tenant, not multi-AZ on RDS (single staging env, cost-tuned).
- No A/B testing framework, no feature flag system.
- No formal eval harness for the agent (the `tests/test_chat.py` suite is integration-level: 6 async tests for the API surface).
- No upstream data scraping — the ingestion pipeline assumes the OpenReferral-ish source tables are already populated.
- No mobile native apps — the SPA is responsive but web-only.
- No map UI — addresses render as text.

---

## 9. Repository pointers

| Repo | Local path |
|------|------|
| `shelter-chat-api` | `c:\Anshul\code\projects\shelter-chat-api` |
| `shelter-mcp-server` | `c:\Anshul\code\projects\shelter-mcp-server` |
| `shelter-search` (frontend + `/mockups`) | `c:\Anshul\code\projects\shelter-search` |
| `ingestion-pipeline` | `c:\Anshul\code\projects\ingestion-pipeline` |
| `shelter-infra` | `c:\Anshul\code\projects\shelter-infra` |

GitHub org: `Anchellon`. AWS account: `746669221991`, region `us-east-1`, deployed environment: `Navigator-Staging`.

---

## 10. Interview prep — request lifecycle

> "Walk me through what happens when a navigator sends a message."

### Happy path: new search, no interrupts

1. **Navigator types** a message ("need shelter for an LGBTQ teen on Saturday morning") and hits send.
2. **React SPA** (`api.ts`) opens a `POST /api/v1/chat` SSE connection with a Bearer token (Auth0 access token from `getAccessTokenSilently`). Redux sets `isStreaming: true`.
3. **FastAPI** validates the JWT via `require_user` (RS256, JWKS cached in memory). Derives `thread_id` from `conversation_id` (UUID from prior turn) or generates a new one.
4. **`astream_events()`** starts on the compiled LangGraph graph with the new `HumanMessage` appended to state.
5. **`guardrails` node** — sends the message to NeMo Guardrails (Claude Haiku). If it passes scope/safety checks, the original message flows through unchanged. 30s timeout; on timeout or exception it **fails open** and passes the message anyway.
6. **`resolve_intent` node** — LLM classifies the message into one of 8 intents. This message → `new_search`.
7. **`classify_groups` node** — LLM extracts structured `Group` records: `{what: "shelter", who: "LGBTQ teen", where: "San Francisco", when: "Saturday morning", open_now: false}` plus a `client_context` block capturing demographics (age, housing, gender, etc.). SSE event `groups_identified` fires; **frontend renders group chips in the chat pane**.
8. **`geo_check` node** — calls MCP `geocode_location` for each group's `where`. "San Francisco" resolves to a lat/lng inside the SF bounding box → group kept.
9. **`intake` node** — calls MCP `list_categories` and `list_eligibilities`. LLM maps `what: "shelter"` → `["sfsg-shelter"]`, `who: "LGBTQ teen"` → `["Youth (below 21 years old)", "LGBTQ+"]`. No gaps → no interrupt. Groups now have `categories` and `eligibilities` populated.
10. **`search_per_group` node** — calls MCP `search_services({query: "shelter LGBTQ teen", categories: ["sfsg-shelter"], eligibilities: [...], lat: 37.77, lng: -122.42, when: "Saturday morning"})`. MCP runs cosine similarity against pgvector, applies tag nudges and schedule filter, returns up to 50 results. Then calls `get_service_details_batch` on the top 5 to backfill address/phone/org name.
11. **`format_results` node** — LLM reads the top services' embedding text and writes a 1–2 sentence rationale per group explaining why these services match. Emits `formatted: {"1": {rationale: "...", service_ids: [123, 456]}}`.
12. **`format_complete` SSE event** fires — frontend renders the ResultsPane with service cards. Backend creates a `referrals` row in Postgres and injects a synthetic `AIMessage` checkpoint marker (`id=f"referral_{uuid}"`) via `aupdate_state` so conversation reload reconstructs the exact turn.
13. **`finish` SSE event** — frontend sets `isStreaming: false`.

### With an intake interrupt

After step 9, if the LLM can't confidently map a category or eligibility (e.g., "need help" with no specifics), `intake` calls `langgraph.types.interrupt({group_id, group_label, steps: [...]})`. The stream **returns early** with an `intake_request` SSE event. The frontend renders a multi-step intake wizard. Navigator selects answers → `POST /api/v1/resume` with `{conversation_id, action: "submit", answers: {...}}` → graph resumes from the interrupt point and continues to `search_per_group`.

---

## 11. Interview prep — failure handling

| What breaks | What happens | Handled where |
|-------------|-------------|---------------|
| Guardrails timeout (>30s) | **Fails open** — message passes through unchanged | `guardrails/node.py` try/except with timeout |
| Guardrails blocks input | Returns refusal `AIMessage`, graph ends, `finish` event fires | NeMo policy in `config/prompts.co` |
| All groups outside SF bounding box | `geo_check` drops them all, emits an `AIMessage` explaining the constraint, graph ends cleanly | `geo_check.py` checks remaining groups count |
| MCP server unreachable | Tool calls raise; exception propagates to `runner.py`; `error` SSE event fires with `errorText`, stream aborts | `runner.py` error handling |
| Navigator cancels intake interrupt | `POST /api/v1/resume` with `action: "cancel"` → `Command(resume=None)` → graph ends gracefully | `resume.py` + LangGraph resume logic |
| Auth0 JWT invalid or expired | 401 before the stream even opens | `require_user` dependency raises `HTTPException(401)` |
| `auth0_domain` not set in env | Auth **disabled** — `require_user` returns `"dev"`. Safe for local dev, never deployed to staging | `auth.py` |
| LLM call fails (API error) | Exception propagates, `error` SSE event, stream aborts. No retry at app level | Node code, propagated through `runner.py` |
| Database (checkpointer) unavailable | `AsyncPostgresSaver` raises on checkpoint write; stream aborts | FastAPI lifespan catches startup failure; mid-stream fails hard |
| CloudFront 60s idle timeout on slow search | **`with_heartbeat()` wrapper** in `runner.py` emits a `_heartbeat` sentinel event every 15s, resetting the idle timer without cancelling in-flight LLM calls | `runner.py` |

---

## 12. Interview prep — tradeoff justifications

> These are the "why not X?" questions you'll get. Lead with the problem you were solving, then the tradeoff.

**Why LangGraph instead of a plain function pipeline?**
The HITL requirement forced it. `interrupt()` lets the graph pause mid-execution, survive a full HTTP round-trip (the frontend collect answers, the navigator POST to `/api/v1/resume`), and resume from the exact node — all with durable state in Postgres. A function pipeline would require hand-rolling that state machine. LangGraph also gave the `AsyncPostgresSaver` checkpointer for free, so conversation history and multi-turn continuity are built in.

**Why MCP instead of calling the search DB directly from the agent?**
Separation of concerns. The chat-api doesn't need to know about pgvector, Bedrock embeddings, or the SQL schema. The MCP server owns all of that, exposes a clean tool surface (7 tools), and can be scaled, replaced, or updated independently. It also means the search logic is reusable by other clients without exposing the DB.

**Why NeMo Guardrails over a simple system prompt filter?**
NeMo uses a declarative Colang policy file (`prompts.co`) that's completely separate from the agent graph. Updating guardrail rules doesn't touch graph code. The 30s fail-open timeout was also a deliberate UX choice — a blocked guardrails call shouldn't stall the navigator's workflow.

**Why HTTP/1.1 forced on CloudFront instead of HTTP/2?**
HTTP/2 multiplexing on CloudFront buffers SSE events rather than streaming them — the client receives chunks in batches, not as they arrive. Forcing HTTP/1.1 on the CloudFront → ALB leg restores standard SSE semantics. This is a non-obvious AWS gotcha, not a general HTTP/2 problem.

**Why SSE over WebSockets?**
The communication pattern is unidirectional — the server streams results to the client; the client sends discrete POST requests (not a continuous stream). SSE is simpler: it works through standard HTTP proxies, auto-reconnects, and doesn't require an upgrade handshake. WebSockets would add complexity with no benefit here.

**Why Boto3 directly in the ingestion pipeline instead of LangChain's `BedrockEmbeddings`?**
LangChain's wrapper has a built-in 4-attempt retry loop. Combined with the pipeline's own exponential-backoff-with-jitter retry, a single quota spike triggered 4 × 4 = 16 attempts per batch. Calling Bedrock via Boto3 directly collapses it to one predictable retry loop, which was also easier to tune for the 10 req/s steady-state quota.

**Why tag nudges (score adjustments) instead of hard tag filters in `search_services`?**
Hard filtering on categories and eligibilities would silently drop services that are semantically correct but not precisely tagged in the database. Tag coverage in OpenReferral-style data is inconsistent. Nudges (–0.10 for category match, –0.05 for eligibility match) let the vector search stay permissive — the right service still surfaces even if its tags are incomplete — while structured intent still outranks untagged results.

**Why no ORM (no SQLAlchemy)?**
The queries are too complex for an ORM to help. The ingestion pipeline's core query is 637 lines of SQL with 12 CTEs, lateral joins, window functions, and CASE-heavy eligibility remapping. Expressing that through an ORM would obscure the intent and fight the schema. For the chat-api, the queries are simpler but raw `psycopg` async keeps the stack minimal with no extra abstraction layer.

**Why Flyway instead of Alembic?**
Flyway is language-agnostic and runs as a standalone Docker container — no Python environment required to run migrations in CI or on a fresh deploy. Migration files are plain SQL, version-controlled, and never edited once applied. Alembic is tightly coupled to SQLAlchemy models, which aren't used here.

**Why a separate `ClientContext` / `case_context` pattern instead of putting demographics on each Group?**
A navigator is often describing multiple people in one conversation ("need shelter for a family with kids, and separately food for a senior"). Demographics often apply to everyone on a case (all are unhoused, all speak Spanish) — the `case_context` captures that once. Where one person differs, `group.client_context` overrides. `effective_context()` merges both at intake time. Without this, the navigator would have to re-specify demographics for every group.

---

## 13. Interview prep — what you'd improve

**If I had more time, here's what I'd tackle first:**

1. **Hard grounding instead of soft.** Right now the "no hallucination" constraint is prompt-only. An evaluator pass that checks the LLM's prose references against the raw tool results (e.g., every service name and address mentioned must appear in `results`) would make it a hard guarantee. The current soft constraint is good enough in practice but not provably safe.

2. **`list_categories` / `list_eligibilities` caching.** These are called fresh on every intake — the data changes perhaps weekly. A short-lived in-process cache (5-minute TTL) would shave one round-trip per search without complicating the architecture.

3. **Formal agent eval harness.** The test suite is 6 integration-level API tests. There's no golden-dataset eval for the agent's actual output quality — no benchmarks for classification accuracy, no grounding checks, no latency profiles. Without evals, regressions are invisible until a navigator complains.

4. **Batched intake HITL across groups.** Currently, each group can trigger its own interrupt separately — a navigator could hit three intake forms in sequence for a three-group message. Batching all clarification questions for the same turn into one interrupt would be a better UX.

5. **Observability at the application layer.** Langfuse, Grafana, and Sentry are wired at the infrastructure sidecar level (OTEL collector), but the chat-api code doesn't emit custom spans, doesn't tag LLM calls with trace IDs, and doesn't record intent classification latency or per-node timing. Adding `@observe()` decorators (Langfuse) to each node would unlock per-node latency breakdowns and LLM cost attribution.

6. **ALB idle timeout for slow searches.** The ALB default is 60 seconds. A cold MCP search (cold pgvector + Bedrock embedding) can approach that. The heartbeat in `runner.py` solves the CloudFront side, but the ALB itself can still close the connection. Bumping the ALB idle timeout to 120s is a one-line CDK change that's flagged in the infra repo but not yet applied.

7. **`resolve_intent` fallback quality.** When a message is ambiguous between `refine` and `new_search`, the classifier can misfire. Adding a confidence threshold with a clarification fallback (already wired as an intent) and a small golden-set evaluation for intent classification accuracy would catch this systematically.

---

## 14. Interview prep — numbers cheat sheet

> Things to have ready. Mark ones you need to verify from the live system.

| Metric | Value | Source |
|--------|-------|--------|
| LangGraph nodes | 13 | `app/agent/graph.py` |
| Valid intents | 8 | `resolve_intent.py` |
| MCP tools | 7 | `shelter-mcp-server/app/tools/` |
| SSE event types | 13 | `runner.py` + `chat.py` |
| Max search results returned | 50 per group | `search_services` SQL LIMIT |
| Top results enriched with full details | 5 per group | `search_per_group.py` |
| Max converse tool iterations | 3 | `converse.py` |
| Heartbeat interval | 15s | `runner.py` `with_heartbeat()` |
| Guardrails timeout | 30s | `guardrails/node.py` |
| SF bounding box | lat 37.63–37.84 / lng –122.52 to –122.35 | `geo_check.py` |
| Embedding dimensions | 1024 | Bedrock Titan v2 |
| Ingestion embedding batch size | 25 | `ingestion-pipeline` config |
| Ingestion Bedrock rate limit | 10 req/s | `ingestion-pipeline` |
| Flyway migrations | 8 | `migrations/V1–V8` |
| DB connection pool (MCP) | 2–10 connections | `app/tools/db.py` |
| Fargate task sizes | chat-api: 1 vCPU / 2 GB; MCP: 0.5 vCPU / 1 GB | `shelter-infra` |
| Fargate autoscale | 1–4 tasks at 70% CPU/mem | `shelter-infra` |
| RDS instance (staging) | t3.micro, PostgreSQL 16 | `shelter-infra` |
| CloudFormation stacks | 6 per environment | `shelter-infra/app.py` |
| CDK repos | 5 total (network, db, mcp, agent, frontend + ingestion) | `shelter-infra` |
| *(verify before interview)* Number of services in DB | — | run `SELECT COUNT(*) FROM service_snapshots` |
| *(verify before interview)* Active navigators | — | check Auth0 tenant or ask |
| *(verify before interview)* Typical search latency (p50) | — | Grafana dashboard |

---

## 15. Interview prep — testing strategy

**What exists:**

- `shelter-chat-api/tests/test_chat.py` — 6 async integration tests hitting the actual FastAPI app (real graph, mocked MCP responses). Covers: new conversation, multi-turn continuation, intake interrupt + resume, cancel, referral creation, auth bypass (`"dev"` user).
- `shelter-mcp-server/tests/test_search.py` — unit/integration tests for `search_services` and `search_by_name` against a real test DB.
- `shelter-search` CI — TypeScript lint + Vite build check on every PR (no component tests, no E2E).

**What doesn't exist (honest gaps):**
- No golden-dataset eval for agent output quality (intent classification accuracy, grounding correctness, rationale quality).
- No load/latency tests.
- No E2E browser tests for the React SPA.
- No contract tests between the SSE event schema the backend emits and the frontend parser expects.

**How you'd explain the testing philosophy:**
The agent's correctness is hard to unit-test because it's LLM-dependent. The integration tests verify the API contract (events fire, state persists, auth enforces) but not the semantic quality of results. That's a deliberate gap — correctness at the LLM layer needs an eval harness with a ground-truth dataset of "navigator says X → expected groups/services", which is future work.
