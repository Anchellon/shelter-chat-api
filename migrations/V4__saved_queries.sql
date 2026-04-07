CREATE TABLE saved_queries (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    group_id    INTEGER NOT NULL,
    title       TEXT NOT NULL,
    group_data  JSONB NOT NULL,
    rationale   TEXT,
    service_ids INTEGER[] NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_saved_queries_user_id
ON saved_queries (user_id, created_at DESC);
