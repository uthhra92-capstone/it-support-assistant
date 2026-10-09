"""SQLite connection management, schema creation and seeding from JSON files."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    department    TEXT NOT NULL,
    title         TEXT NOT NULL,
    location      TEXT NOT NULL,
    manager       TEXT,
    status        TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    assigned_devices TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id     TEXT PRIMARY KEY,
    employee_id   TEXT NOT NULL REFERENCES employees(employee_id),
    category      TEXT NOT NULL,
    subject       TEXT NOT NULL,
    description   TEXT NOT NULL,
    priority      TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    status        TEXT NOT NULL CHECK (status IN ('open', 'in_progress', 'on_hold', 'resolved', 'closed')),
    assigned_team TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    resolution_notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_employee ON tickets(employee_id);

CREATE TABLE IF NOT EXISTS kb_articles (
    article_id    TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    category      TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '[]',
    content       TEXT NOT NULL,
    last_updated  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_status (
    system_name   TEXT PRIMARY KEY,
    status        TEXT NOT NULL CHECK (status IN ('operational', 'degraded', 'outage', 'maintenance')),
    message       TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""


class Database:
    """Thin wrapper that hands out short-lived connections (safe across Streamlit threads)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self, seed_dir: Path | None = None) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            if seed_dir is not None and self._is_empty(conn):
                self._seed(conn, seed_dir)

    @staticmethod
    def _is_empty(conn: sqlite3.Connection) -> bool:
        return conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0

    @staticmethod
    def _load(seed_dir: Path, name: str) -> list[dict]:
        path = seed_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Seed file missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _seed(self, conn: sqlite3.Connection, seed_dir: Path) -> None:
        employees = self._load(seed_dir, "employees.json")
        tickets = self._load(seed_dir, "tickets.json")
        articles = self._load(seed_dir, "knowledge_base.json")
        statuses = self._load(seed_dir, "system_status.json")

        conn.executemany(
            """INSERT INTO employees VALUES
               (:employee_id, :name, :email, :department, :title, :location, :manager, :status, :assigned_devices)""",
            [{**e, "assigned_devices": json.dumps(e.get("assigned_devices", []))} for e in employees],
        )
        conn.executemany(
            """INSERT INTO tickets VALUES
               (:ticket_id, :employee_id, :category, :subject, :description, :priority, :status,
                :assigned_team, :created_at, :updated_at, :resolution_notes)""",
            [{"resolution_notes": None, **t} for t in tickets],
        )
        conn.executemany(
            """INSERT INTO kb_articles VALUES
               (:article_id, :title, :category, :tags, :content, :last_updated)""",
            [{**a, "tags": json.dumps(a.get("tags", []))} for a in articles],
        )
        conn.executemany(
            "INSERT INTO system_status VALUES (:system_name, :status, :message, :updated_at)",
            statuses,
        )
        logger.info(
            "Seeded database: %d employees, %d tickets, %d KB articles, %d systems",
            len(employees), len(tickets), len(articles), len(statuses),
        )

    def reset(self, seed_dir: Path) -> None:
        """Drop all data and re-seed. Used by scripts/init_db.py --reset."""
        with self.connect() as conn:
            for table in ("tickets", "employees", "kb_articles", "system_status"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.initialize(seed_dir)
