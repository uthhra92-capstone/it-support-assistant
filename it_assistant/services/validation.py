"""Input validation rules applied before any tool touches the database."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

EMPLOYEE_ID_RE = re.compile(r"^EMP\d{4}$")
TICKET_ID_RE = re.compile(r"^INC\d{6}$")
TICKET_ID_IN_TEXT_RE = re.compile(r"\bINC\d{6}\b")

TICKET_CATEGORIES = (
    "vpn", "network", "hardware", "software", "email", "account_access", "printer", "other",
)
TICKET_PRIORITIES = ("low", "medium", "high", "critical")
TICKET_STATUS_FILTERS = ("open", "closed", "all")

MIN_SUBJECT_CHARS = 5
MAX_SUBJECT_CHARS = 120
MIN_DESCRIPTION_CHARS = 15
MAX_DESCRIPTION_CHARS = 2000


def normalize_employee_id(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().upper().replace(" ", "")
    return candidate if EMPLOYEE_ID_RE.match(candidate) else None


def normalize_ticket_id(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().upper().replace(" ", "")
    return candidate if TICKET_ID_RE.match(candidate) else None


@dataclass
class TicketDraft:
    category: str
    subject: str
    description: str
    priority: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "subject": self.subject,
            "description": self.description,
            "priority": self.priority,
        }


@dataclass
class ValidationResult:
    draft: TicketDraft | None = None
    missing_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.draft is not None and not self.missing_fields and not self.errors


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def validate_ticket_request(args: dict[str, Any]) -> ValidationResult:
    """Validate the arguments the LLM supplied for create_ticket.

    The LLM's arguments are never trusted as-is: every field is re-checked here.
    """
    result = ValidationResult()

    category = _clean_str(args.get("category")).lower()
    subject = _clean_str(args.get("subject"))
    description = _clean_str(args.get("description"))
    priority = _clean_str(args.get("priority")).lower()

    if not category:
        result.missing_fields.append("category")
    elif category not in TICKET_CATEGORIES:
        result.errors.append(f"category must be one of {list(TICKET_CATEGORIES)}")

    if not subject:
        result.missing_fields.append("subject")
    elif not MIN_SUBJECT_CHARS <= len(subject) <= MAX_SUBJECT_CHARS:
        result.errors.append(
            f"subject must be between {MIN_SUBJECT_CHARS} and {MAX_SUBJECT_CHARS} characters"
        )

    if not description:
        result.missing_fields.append("description")
    elif len(description) < MIN_DESCRIPTION_CHARS:
        result.errors.append(
            "description is too short - ask the user what is happening, since when, "
            "and any error message they see"
        )
    elif len(description) > MAX_DESCRIPTION_CHARS:
        result.errors.append(f"description must be at most {MAX_DESCRIPTION_CHARS} characters")

    if not priority:
        result.missing_fields.append("priority")
    elif priority not in TICKET_PRIORITIES:
        result.errors.append(f"priority must be one of {list(TICKET_PRIORITIES)}")

    if not result.missing_fields and not result.errors:
        result.draft = TicketDraft(category, subject, description, priority)
    return result
