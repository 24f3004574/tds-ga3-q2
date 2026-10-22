import os
import threading

import psycopg

SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        principal_hash TEXT NOT NULL,
        context_id TEXT NOT NULL,
        batch_id TEXT,
        state TEXT NOT NULL,
        task_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE INDEX IF NOT EXISTS idx_tasks_owner_created
        ON tasks (principal_hash, created_at DESC)""",
    """CREATE TABLE IF NOT EXISTS message_dedup (
        principal_hash TEXT NOT NULL,
        message_id TEXT NOT NULL,
        message_hash TEXT NOT NULL,
        task_id TEXT NOT NULL REFERENCES tasks(task_id),
        response_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (principal_hash, message_id)
    )""",
    """CREATE TABLE IF NOT EXISTS decision_cache (
        package_hash TEXT PRIMARY KEY,
        canonical_package JSONB NOT NULL,
        decision_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
]

_schema_ready = False
_schema_lock = threading.Lock()


def database_url():
    value = os.getenv("POSTGRES_URL")
    if not value:
        raise RuntimeError("Database configuration is missing")
    return value


def init_db():
    """Create tables once per process. Safe to call from many requests/instances."""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with psycopg.connect(database_url(), connect_timeout=8) as connection:
            with connection.cursor() as cursor:
                # Serialize schema creation across serverless instances.
                cursor.execute("SELECT pg_advisory_xact_lock(727274001)")
                for statement in SCHEMA_STATEMENTS:
                    cursor.execute(statement)
        _schema_ready = True


def connect_db():
    init_db()
    return psycopg.connect(database_url(), connect_timeout=8)
