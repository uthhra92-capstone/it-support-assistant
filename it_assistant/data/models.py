"""Domain models returned by repositories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Employee:
    employee_id: str
    name: str
    email: str
    department: str
    title: str
    location: str
    manager: str | None
    status: str
    assigned_devices: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    employee_id: str
    category: str
    subject: str
    description: str
    priority: str
    status: str
    assigned_team: str
    created_at: str
    updated_at: str
    resolution_notes: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status in {"open", "in_progress", "on_hold"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeArticle:
    article_id: str
    title: str
    category: str
    tags: list[str]
    content: str
    last_updated: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SystemStatus:
    system_name: str
    status: str
    message: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
