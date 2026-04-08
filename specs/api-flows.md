# API Flows — shelter-chat-api

This document describes the full user journey and maps each step to the API endpoints
that support it. Use this as the source of truth when building or reviewing features.

---

## User journey overview

```
1. User starts a chat describing their need
2. Agent may ask clarifying questions (HITL intake)
3. Agent searches services and builds a referral
4. Referral is auto-saved to DB and streamed to the frontend
5. User can continue chatting to get more referrals in the same session
6. User explicitly saves a referral → it appears in their Collections page
7. User can start a new chat at any time
8. User can browse their past chat sessions (Chat History)
9. User can view and manage saved referrals (Collections)
```

---

## Step-by-step flows

### 1. Start a conversation

The user types their need. A new `conversation_id` is created if none is provided.

```
POST /api/v1/chat
Body: { message, conversation_id? (omit to start new), current_time? }
Auth: Bearer token required
Returns: text/event-stream (SSE)
```

**SSE events in order:**
| event | meaning |
|-------|---------|
| `text-start` | message begins |
| `tool-start` / `tool-end` | agent is calling MCP tools (visible status) |
| `groups_identified` | agent has parsed need groups |
| `format_complete` | results ready; includes `formatted`, `groups`, `referral_id` |
| `text-end` | message ends |
| `finish` | stream complete |

The `X-Conversation-Id` response header always carries the `conversation_id` (useful when starting a new conversation so the frontend can track it).

---

### 2. HITL intake (clarifying questions)

If the agent cannot map a group to categories or eligibilities, it pauses and emits:

```
SSE event: intake_request
  { type, group_id, group_label, steps: [{ dimension, type, question, options }] }
```

The stream ends here. The frontend renders the form and submits answers:

```
POST /api/v1/chat/resume
Body: { conversation_id, action: "submit" | "cancel", answers: { what?, who?, where? } }
Auth: Bearer token required
Returns: text/event-stream (same SSE event set as /chat)
```

If the user cancels, send `action: "cancel"`. The agent discards that group and moves on.
If another group also has gaps, a second `intake_request` will fire after this resume.

---

### 3 & 4. Agent searches and creates a referral

This happens automatically inside the agent pipeline — no separate client call needed.

When `format_complete` fires, the referral has already been saved to the database.

```
SSE event: format_complete
  {
    type: "format_complete",
    formatted: { "1": { rationale, service_ids }, ... },
    groups: [ Group, ... ],
    referral_id: "<uuid>"
  }
```

The frontend uses `referral_id` to link the displayed result to the DB record.

---

### 5. Continue chatting / get more referrals

Send another message with the **same `conversation_id`**. The agent resumes from the
existing LangGraph checkpoint. A new referral is created for each `format_complete` event.

```
POST /api/v1/chat
Body: { conversation_id: "<existing id>", message: "..." }
```

---

### 6. Save a referral (Collections)

Every referral is auto-created in the DB (unsaved by default, `saved = false`).
The user must explicitly save it to add it to their Collections page.

```
PATCH /api/v1/referrals/{referral_id}/save
Auth: Bearer token required
Returns: { id, saved: true }
```

> **Note:** There is currently no "unsave" endpoint. Add one when needed:
> `PATCH /api/v1/referrals/{referral_id}/unsave`

---

### 7. Start a new chat

Omit `conversation_id` in the chat request. The backend generates a new UUID.
The frontend reads it from the `X-Conversation-Id` response header.

---

### 8. Chat history

List all past conversations for the current user:

```
GET /api/v1/conversations
Auth: Bearer token required
Returns: { conversations: [{ id, title }] }
```

Load a full conversation (messages + referrals):

```
GET /api/v1/conversations/{conversation_id}
Auth: Bearer token required
Returns:
  {
    id,
    messages: [{ id, role: "user"|"assistant", type: "text", content }],
    groups: [ Group ],
    formatted: { ... },
    referrals: [{ id, title, saved, groups, created_at }]
  }
```

---

### 9. Collections page

List all **saved** referrals for the current user:

```
GET /api/v1/referrals
Auth: Bearer token required
Returns:
  {
    referrals: [{ id, thread_id, title, saved, groups (with service_count), created_at }]
  }
```

> Note: `service_ids` is stripped from each group in the list response; `service_count`
> is returned instead. Use `GET /api/v1/referrals/{id}` to get the full group data.

Get a single saved referral (full detail):

```
GET /api/v1/referrals/{referral_id}
Auth: Bearer token required
Returns: { id, thread_id, title, saved, groups (full), created_at }
```

Delete a referral:

```
DELETE /api/v1/referrals/{referral_id}
Auth: Bearer token required
Returns: 204 No Content
```

---

### Fetch service details (for rendering results)

The `service_ids` in a referral are IDs from the shelter-search MCP server.
To render them, batch-fetch the details:

```
POST /api/v1/services/batch
Body: { service_ids: [123, 456, ...] }
Auth: Bearer token required
Returns: { services: [ ServiceDetail, ... ] }
```

---

## Full endpoint reference

| Method | Path | Purpose | Status |
|--------|------|---------|--------|
| `POST` | `/api/v1/chat` | Start or continue a conversation | ✅ built |
| `POST` | `/api/v1/chat/resume` | Resume after HITL intake interrupt | ✅ built |
| `GET` | `/api/v1/conversations` | List chat history | ✅ built |
| `GET` | `/api/v1/conversations/{id}` | Full conversation + referrals | ✅ built |
| `POST` | `/api/v1/services/batch` | Fetch service details by id list | ✅ built |
| `POST` | `/api/v1/referrals` | Create referral manually | ✅ built |
| `PATCH` | `/api/v1/referrals/{id}/save` | Save a referral to Collections | ✅ built |
| `GET` | `/api/v1/referrals` | List saved referrals (Collections) | ✅ built |
| `GET` | `/api/v1/referrals/{id}` | Get single referral (full detail) | ✅ built |
| `DELETE` | `/api/v1/referrals/{id}` | Delete a referral | ✅ built |
| `PATCH` | `/api/v1/referrals/{id}/unsave` | Remove from Collections | ❌ to build |
| `PATCH` | `/api/v1/referrals/{id}/rename` | Rename a referral | ❌ to build |
| `PATCH` | `/api/v1/conversations/{id}/rename` | Rename a conversation | ❌ to build |
| `GET` | `/health` | Liveness / readiness check | ✅ built |

---

## Features to build

### Unsave a referral

Remove a referral from Collections without deleting it. Sets `saved = false`.

```
PATCH /api/v1/referrals/{referral_id}/unsave
Auth: Bearer token required
Returns: { id, saved: false }
```

Implementation: mirror `save` endpoint — `UPDATE referrals SET saved = FALSE WHERE id = %s AND user_id = %s`.

---

### Rename a referral

Allow the user to give a referral a custom title instead of the auto-generated one.

```
PATCH /api/v1/referrals/{referral_id}/rename
Auth: Bearer token required
Body: { title: string }
Returns: { id, title }
```

Validation: `title` must be non-empty, max 120 chars.
Implementation: `UPDATE referrals SET title = %s WHERE id = %s AND user_id = %s`.

---

### Rename a conversation

Allow the user to rename a conversation in chat history.

```
PATCH /api/v1/conversations/{conversation_id}/rename
Auth: Bearer token required
Body: { title: string }
Returns: { id, title }
```

Validation: `title` must be non-empty, max 80 chars (matches the existing truncation in `save_conversation_summary`).
Implementation: `UPDATE conversation_summaries SET title = %s WHERE thread_id = %s AND user_id = %s`.

---

### Pagination for list endpoints

Both `GET /conversations` and `GET /referrals` currently hard-cap at 50 rows with no way to page.
Add cursor-based pagination using `created_at` + `id` as the cursor.

**Conversations:**
```
GET /api/v1/conversations?limit=20&before=<cursor>
Returns:
  {
    conversations: [{ id, title }],
    next_cursor: "<cursor>" | null
  }
```

**Referrals (Collections):**
```
GET /api/v1/referrals?limit=20&before=<cursor>
Returns:
  {
    referrals: [...],
    next_cursor: "<cursor>" | null
  }
```

Cursor encoding: base64 of `"{created_at}_{id}"`. Query uses `WHERE created_at < cursor_ts OR (created_at = cursor_ts AND id < cursor_id)` with `ORDER BY created_at DESC, id DESC LIMIT limit+1` (fetch one extra to determine if there's a next page).
