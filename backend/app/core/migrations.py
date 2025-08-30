"""
Schema-less migration helper to ensure new columns exist.
Currently adds tickets.resolution_code if missing (SQLite/Postgres).
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _column_exists_sqlite(engine: Engine, table: str, column: str) -> bool:
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        for r in rows:
            if r[1] == column:
                return True
        return False


def _column_exists_postgres(engine: Engine, table: str, column: str) -> bool:
    with engine.connect() as conn:
        rows = conn.execute(text(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = :t AND column_name = :c
            """
        ), {"t": table, "c": column}).fetchone()
        return rows is not None


def add_column_if_missing(engine: Engine, table: str, column: str, coldef_sql: str) -> None:
    url = str(engine.url)
    is_sqlite = url.startswith("sqlite:")
    exists = _column_exists_sqlite(engine, table, column) if is_sqlite else _column_exists_postgres(engine, table, column)
    if exists:
        return
    ddl = f"ALTER TABLE {table} ADD COLUMN {column} {coldef_sql}"
    with engine.begin() as conn:
        try:
            conn.execute(text(ddl))
        except Exception:
            pass


def ensure_ticket_columns(engine: Engine) -> None:
    # resolution_code stored as VARCHAR/TEXT
    add_column_if_missing(engine, "tickets", "resolution_code", "VARCHAR(255)")

