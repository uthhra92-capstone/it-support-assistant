"""Repository classes - the only code that issues SQL."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from it_assistant.data.database import Database
from it_assistant.data.models import Employee, KnowledgeArticle, SystemStatus, Ticket

OPEN_STATUSES = ("open", "in_progress", "on_hold")

# Which resolver team receives a new ticket, by category.
TEAM_BY_CATEGORY = {
    "vpn": "Network Operations",
    "network": "Network Operations",
    "hardware": "Desktop Support",
    "software": "Application Support",
    "email": "Messaging Team",
    "account_access": "Identity & Access Management",
    "printer": "Desktop Support",
    "other": "IT Service Desk",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class EmployeeRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, employee_id: str) -> Employee | None:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["assigned_devices"] = json.loads(data["assigned_devices"] or "[]")
        return Employee(**data)


class TicketRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, ticket_id: str) -> Ticket | None:
        with self._db.connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        return Ticket(**dict(row)) if row else None

    def exists(self, ticket_id: str) -> bool:
        return self.get(ticket_id) is not None

    def list_for_employee(
        self,
        employee_id: str,
        status: str | None = None,
        keyword: str | None = None,
        limit: int = 10,
    ) -> list[Ticket]:
        sql = "SELECT * FROM tickets WHERE employee_id = ?"
        params: list[object] = [employee_id]
        if status == "open":
            sql += f" AND status IN ({','.join('?' * len(OPEN_STATUSES))})"
            params.extend(OPEN_STATUSES)
        elif status == "closed":
            sql += " AND status IN ('resolved', 'closed')"
        if keyword:
            sql += " AND (LOWER(subject) LIKE ? OR LOWER(description) LIKE ? OR LOWER(category) LIKE ?)"
            like = f"%{keyword.lower()}%"
            params.extend([like, like, like])
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Ticket(**dict(r)) for r in rows]

    def list_open_for_employee(self, employee_id: str) -> list[Ticket]:
        return self.list_for_employee(employee_id, status="open", limit=50)

    def create(
        self,
        employee_id: str,
        category: str,
        subject: str,
        description: str,
        priority: str,
    ) -> Ticket:
        now = utc_now_iso()
        team = TEAM_BY_CATEGORY.get(category, TEAM_BY_CATEGORY["other"])
        with self._db.connect() as conn:
            # BEGIN IMMEDIATE takes the write lock up front so two sessions cannot
            # generate the same ticket number.
            conn.execute("BEGIN IMMEDIATE")
            ticket_id = self._next_ticket_id(conn)
            conn.execute(
                """INSERT INTO tickets (ticket_id, employee_id, category, subject, description,
                                        priority, status, assigned_team, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
                (ticket_id, employee_id, category, subject, description, priority, team, now, now),
            )
        ticket = self.get(ticket_id)
        assert ticket is not None
        return ticket

    @staticmethod
    def _next_ticket_id(conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT MAX(CAST(SUBSTR(ticket_id, 4) AS INTEGER)) FROM tickets WHERE ticket_id LIKE 'INC%'"
        ).fetchone()
        next_number = (row[0] or 100000) + 1
        return f"INC{next_number:06d}"


class KnowledgeBaseRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def all(self) -> list[KnowledgeArticle]:
        with self._db.connect() as conn:
            rows = conn.execute("SELECT * FROM kb_articles ORDER BY article_id").fetchall()
        articles = []
        for row in rows:
            data = dict(row)
            data["tags"] = json.loads(data["tags"] or "[]")
            articles.append(KnowledgeArticle(**data))
        return articles


class SystemStatusRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def list_all(self) -> list[SystemStatus]:
        with self._db.connect() as conn:
            rows = conn.execute("SELECT * FROM system_status ORDER BY system_name").fetchall()
        return [SystemStatus(**dict(r)) for r in rows]
