"""Tool implementations. Every tool returns a ToolResult - tools never raise to the graph."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from it_assistant.config import Settings
from it_assistant.data.database import Database
from it_assistant.data.models import Employee, Ticket
from it_assistant.data.repositories import (
    EmployeeRepository,
    KnowledgeBaseRepository,
    SystemStatusRepository,
    TicketRepository,
)
from it_assistant.services.knowledge_search import KnowledgeSearchIndex
from it_assistant.services.text import jaccard_similarity
from it_assistant.services.validation import (
    TICKET_STATUS_FILTERS,
    TicketDraft,
    normalize_employee_id,
    normalize_ticket_id,
    validate_ticket_request,
)

logger = logging.getLogger(__name__)

# ToolResult.status values
SUCCESS = "success"
NOT_FOUND = "not_found"
NEEDS_INPUT = "needs_input"
DUPLICATE = "possible_duplicate"
CONFIRMATION_REQUIRED = "confirmation_required"
ERROR = "error"


@dataclass
class Source:
    """Provenance record for data retrieved from a local system (shown in the UI)."""

    kind: str  # kb_article | ticket | employee | system_status
    ref: str
    title: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, "title": self.title}


@dataclass
class ToolResult:
    status: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.status == ERROR

    def to_model_content(self) -> str:
        """Serialized payload sent back to the LLM as the tool message."""
        return json.dumps({"status": self.status, "message": self.message, "data": self.data}, default=str)


def _employee_summary(employee: Employee) -> dict[str, Any]:
    return {
        "employee_id": employee.employee_id,
        "name": employee.name,
        "department": employee.department,
        "title": employee.title,
        "location": employee.location,
        "assigned_devices": employee.assigned_devices,
    }


def _ticket_summary(ticket: Ticket) -> dict[str, Any]:
    return {
        "ticket_id": ticket.ticket_id,
        "category": ticket.category,
        "subject": ticket.subject,
        "status": ticket.status,
        "priority": ticket.priority,
        "assigned_team": ticket.assigned_team,
        "created_at": ticket.created_at,
        "updated_at": ticket.updated_at,
        "resolution_notes": ticket.resolution_notes,
    }


class ToolExecutor:
    def __init__(self, db: Database, settings: Settings) -> None:
        self._settings = settings
        self.employees = EmployeeRepository(db)
        self.tickets = TicketRepository(db)
        self.statuses = SystemStatusRepository(db)
        self._kb_repo = KnowledgeBaseRepository(db)
        self._kb_index: KnowledgeSearchIndex | None = None

    # ------------------------------------------------------------------ employees
    def get_employee_profile(self, args: dict[str, Any]) -> ToolResult:
        employee_id = normalize_employee_id(args.get("employee_id"))
        if employee_id is None:
            return ToolResult(
                NEEDS_INPUT,
                "Invalid employee ID format. Expected 'EMP' followed by 4 digits, e.g. EMP1024. "
                "Ask the user to re-check their employee ID.",
            )
        employee = self.employees.get(employee_id)
        if employee is None:
            return ToolResult(NOT_FOUND, f"No employee found with ID {employee_id}.")
        if not employee.is_active:
            return ToolResult(
                ERROR,
                f"Employee {employee_id} is not active. IT support requests for inactive accounts "
                "must go through the employee's manager or HR.",
            )
        return ToolResult(
            SUCCESS,
            f"Employee {employee_id} verified.",
            data={"employee": _employee_summary(employee)},
            sources=[Source("employee", employee.employee_id, employee.name)],
        )

    # ----------------------------------------------------------------- knowledge
    def _index(self) -> KnowledgeSearchIndex:
        if self._kb_index is None:
            self._kb_index = KnowledgeSearchIndex(self._kb_repo.all())
        return self._kb_index

    def search_knowledge_base(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(NEEDS_INPUT, "A non-empty search query is required.")
        hits = self._index().search(
            query, top_k=self._settings.kb_top_k, min_score=self._settings.kb_min_score
        )
        if not hits:
            return ToolResult(
                NOT_FOUND,
                f"No knowledge base article matched '{query}'. Do not invent instructions; "
                "offer to raise a ticket instead.",
            )
        articles = [
            {
                "article_id": h.article.article_id,
                "title": h.article.title,
                "relevance_score": h.score,
                "last_updated": h.article.last_updated,
                "content": h.article.content,
            }
            for h in hits
        ]
        return ToolResult(
            SUCCESS,
            f"Found {len(articles)} relevant article(s).",
            data={"articles": articles},
            sources=[Source("kb_article", h.article.article_id, h.article.title) for h in hits],
        )

    # ------------------------------------------------------------------- tickets
    def lookup_tickets(self, args: dict[str, Any], employee_id: str | None) -> ToolResult:
        if not employee_id:
            return ToolResult(
                NEEDS_INPUT,
                "Employee is not verified yet. Ask the user for their employee ID and call "
                "get_employee_profile before looking up tickets.",
            )

        raw_ticket_id = args.get("ticket_id")
        if raw_ticket_id:
            ticket_id = normalize_ticket_id(raw_ticket_id)
            if ticket_id is None:
                return ToolResult(NEEDS_INPUT, "Ticket IDs look like INC followed by 6 digits, e.g. INC100245.")
            ticket = self.tickets.get(ticket_id)
            # Same message for "missing" and "belongs to someone else" to avoid leaking data.
            if ticket is None or ticket.employee_id != employee_id:
                return ToolResult(NOT_FOUND, f"No ticket {ticket_id} found for employee {employee_id}.")
            return ToolResult(
                SUCCESS,
                "Ticket found.",
                data={"tickets": [_ticket_summary(ticket)]},
                sources=[Source("ticket", ticket.ticket_id, ticket.subject)],
            )

        status = args.get("status") or "all"
        if status not in TICKET_STATUS_FILTERS:
            status = "all"
        keyword = args.get("keyword")
        keyword = keyword.strip() if isinstance(keyword, str) and keyword.strip() else None

        tickets = self.tickets.list_for_employee(
            employee_id, status=None if status == "all" else status, keyword=keyword
        )
        if not tickets and keyword:
            # Keyword was probably too narrow - fall back to all tickets with that status filter.
            tickets = self.tickets.list_for_employee(employee_id, status=None if status == "all" else status)
            note = f"No tickets matched keyword '{keyword}'; showing all {status} tickets instead."
        else:
            note = f"Found {len(tickets)} ticket(s)."
        if not tickets:
            return ToolResult(NOT_FOUND, f"Employee {employee_id} has no {status} tickets on record.")
        return ToolResult(
            SUCCESS,
            note,
            data={"tickets": [_ticket_summary(t) for t in tickets]},
            sources=[Source("ticket", t.ticket_id, t.subject) for t in tickets],
        )

    def prepare_ticket(self, args: dict[str, Any], employee_id: str | None) -> ToolResult:
        """Validate + duplicate-check a ticket request. Does NOT write to the database."""
        if not employee_id:
            return ToolResult(
                NEEDS_INPUT,
                "Employee is not verified yet. Ask the user for their employee ID and call "
                "get_employee_profile before creating a ticket.",
            )
        validation = validate_ticket_request(args)
        if not validation.ok:
            return ToolResult(
                NEEDS_INPUT,
                "The ticket request is incomplete. Ask the user for the missing details.",
                data={"missing_fields": validation.missing_fields, "problems": validation.errors},
            )
        draft = validation.draft
        assert draft is not None

        if not args.get("confirmed_not_duplicate"):
            duplicate = self.find_duplicate(employee_id, draft)
            if duplicate is not None:
                return ToolResult(
                    DUPLICATE,
                    "The employee already has an open ticket that looks like the same issue. "
                    "Share it with the user and ask whether this is a different problem before "
                    "creating another ticket.",
                    data={"existing_ticket": _ticket_summary(duplicate)},
                    sources=[Source("ticket", duplicate.ticket_id, duplicate.subject)],
                )

        return ToolResult(
            CONFIRMATION_REQUIRED,
            "Ticket draft is valid but has NOT been created. Summarise the draft for the user and "
            "ask them to confirm (yes/no).",
            data={"draft": draft.to_dict()},
        )

    def find_duplicate(self, employee_id: str, draft: TicketDraft) -> Ticket | None:
        threshold = self._settings.duplicate_similarity_threshold
        draft_text = f"{draft.subject} {draft.description}"
        best: tuple[float, Ticket] | None = None
        for ticket in self.tickets.list_open_for_employee(employee_id):
            similarity = jaccard_similarity(draft_text, f"{ticket.subject} {ticket.description}")
            if ticket.category == draft.category:
                similarity += 0.25  # same category is a strong duplicate signal
            if similarity >= threshold and (best is None or similarity > best[0]):
                best = (similarity, ticket)
        return best[1] if best else None

    def commit_ticket(self, draft: dict[str, Any], employee_id: str) -> ToolResult:
        """Create the ticket after explicit user confirmation."""
        validation = validate_ticket_request(draft)
        if not validation.ok or validation.draft is None:
            return ToolResult(ERROR, "Stored ticket draft is no longer valid; please start again.")
        employee = self.employees.get(employee_id)
        if employee is None or not employee.is_active:
            return ToolResult(ERROR, f"Employee {employee_id} is no longer valid.")
        d = validation.draft
        try:
            ticket = self.tickets.create(employee_id, d.category, d.subject, d.description, d.priority)
        except Exception:  # noqa: BLE001 - surface any DB failure as a tool error
            logger.exception("Ticket creation failed for %s", employee_id)
            return ToolResult(ERROR, "The ticketing database is unavailable. The ticket was NOT created.")
        logger.info("Created ticket %s for %s", ticket.ticket_id, employee_id)
        return ToolResult(
            SUCCESS,
            f"Ticket {ticket.ticket_id} created.",
            data={"ticket": _ticket_summary(ticket)},
            sources=[Source("ticket", ticket.ticket_id, ticket.subject)],
        )

    # ------------------------------------------------------------- system status
    def check_system_status(self, args: dict[str, Any]) -> ToolResult:
        statuses = self.statuses.list_all()
        name = args.get("system_name")
        if isinstance(name, str) and name.strip():
            needle = name.strip().lower()
            matches = [s for s in statuses if needle in s.system_name.lower() or s.system_name.lower() in needle]
            if not matches:
                return ToolResult(
                    NOT_FOUND,
                    f"No monitored system called '{name}'.",
                    data={"known_systems": [s.system_name for s in statuses]},
                )
            statuses = matches
        return ToolResult(
            SUCCESS,
            f"Status for {len(statuses)} system(s).",
            data={"systems": [s.to_dict() for s in statuses]},
            sources=[Source("system_status", s.system_name, f"{s.system_name}: {s.status}") for s in statuses],
        )
