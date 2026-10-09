"""Unit tests for tools, validation and search (no LLM involved)."""

from it_assistant.services.knowledge_search import KnowledgeSearchIndex
from it_assistant.services.validation import (
    normalize_employee_id,
    validate_ticket_request,
)
from it_assistant.tools.executor import (
    CONFIRMATION_REQUIRED,
    DUPLICATE,
    ERROR,
    NEEDS_INPUT,
    NOT_FOUND,
    SUCCESS,
)

VALID_TICKET = {
    "category": "email",
    "subject": "Cannot send emails with attachments",
    "description": "Since this morning Outlook fails to send any email with an attachment larger than 1 MB.",
    "priority": "medium",
}


# ------------------------------------------------------------------- validation
def test_normalize_employee_id():
    assert normalize_employee_id(" emp1024 ") == "EMP1024"
    assert normalize_employee_id("EMP 1024") == "EMP1024"
    assert normalize_employee_id("1024") is None
    assert normalize_employee_id(None) is None


def test_ticket_validation_reports_missing_fields():
    result = validate_ticket_request({"category": "vpn"})
    assert not result.ok
    assert set(result.missing_fields) == {"subject", "description", "priority"}


def test_ticket_validation_rejects_bad_enum_and_short_description():
    result = validate_ticket_request({**VALID_TICKET, "priority": "urgent", "description": "broken"})
    assert not result.ok
    assert len(result.errors) == 2


# ----------------------------------------------------------------------- search
def test_knowledge_search_finds_vpn_password_article(executor):
    result = executor.search_knowledge_base({"query": "How do I reset my VPN password?"})
    assert result.status == SUCCESS
    assert result.data["articles"][0]["article_id"] == "KB-001"


def test_knowledge_search_no_match(executor):
    result = executor.search_knowledge_base({"query": "quantum chromodynamics lattice"})
    assert result.status == NOT_FOUND


def test_bm25_empty_index():
    assert KnowledgeSearchIndex([]).search("vpn") == []


# -------------------------------------------------------------------- employees
def test_employee_lookup(executor):
    assert executor.get_employee_profile({"employee_id": "emp1024"}).status == SUCCESS
    assert executor.get_employee_profile({"employee_id": "EMP9999"}).status == NOT_FOUND
    assert executor.get_employee_profile({"employee_id": "bob"}).status == NEEDS_INPUT
    assert executor.get_employee_profile({"employee_id": "EMP1006"}).status == ERROR  # inactive


# ---------------------------------------------------------------------- tickets
def test_ticket_lookup_requires_verified_employee(executor):
    assert executor.lookup_tickets({}, employee_id=None).status == NEEDS_INPUT


def test_ticket_lookup_by_keyword(executor):
    result = executor.lookup_tickets({"keyword": "laptop"}, employee_id="EMP1024")
    assert result.status == SUCCESS
    assert result.data["tickets"][0]["ticket_id"] == "INC100231"


def test_ticket_lookup_cannot_read_other_employees_ticket(executor):
    # INC100245 belongs to EMP1001
    result = executor.lookup_tickets({"ticket_id": "INC100245"}, employee_id="EMP1024")
    assert result.status == NOT_FOUND
    assert "data" not in result.to_model_content() or result.data == {}


def test_ticket_lookup_open_filter(executor):
    result = executor.lookup_tickets({"status": "open"}, employee_id="EMP1024")
    assert [t["ticket_id"] for t in result.data["tickets"]] == ["INC100231"]


def test_prepare_ticket_needs_input_when_incomplete(executor):
    result = executor.prepare_ticket({"category": "vpn", "priority": "high"}, employee_id="EMP1024")
    assert result.status == NEEDS_INPUT
    assert "description" in result.data["missing_fields"]


def test_prepare_ticket_requires_confirmation_and_does_not_write(executor):
    before = len(executor.tickets.list_for_employee("EMP1024"))
    result = executor.prepare_ticket(VALID_TICKET, employee_id="EMP1024")
    assert result.status == CONFIRMATION_REQUIRED
    assert len(executor.tickets.list_for_employee("EMP1024")) == before


def test_prepare_ticket_detects_duplicate(executor):
    draft = {
        "category": "vpn",
        "subject": "VPN keeps disconnecting",
        "description": "GlobalProtect VPN disconnects every few minutes when I work from home.",
        "priority": "high",
    }
    result = executor.prepare_ticket(draft, employee_id="EMP1001")
    assert result.status == DUPLICATE
    assert result.data["existing_ticket"]["ticket_id"] == "INC100245"

    # User said it is a different problem -> allowed.
    forced = executor.prepare_ticket({**draft, "confirmed_not_duplicate": True}, employee_id="EMP1001")
    assert forced.status == CONFIRMATION_REQUIRED


def test_commit_ticket_generates_sequential_id(executor):
    first = executor.commit_ticket(VALID_TICKET, "EMP1024")
    second = executor.commit_ticket({**VALID_TICKET, "category": "printer"}, "EMP1024")
    assert first.status == SUCCESS and second.status == SUCCESS
    first_id = first.data["ticket"]["ticket_id"]
    assert first_id == "INC100246"
    assert second.data["ticket"]["ticket_id"] == "INC100247"
    assert executor.tickets.get(first_id).status == "open"
    assert executor.tickets.get(first_id).assigned_team == "Messaging Team"


def test_system_status_filter(executor):
    result = executor.check_system_status({"system_name": "vpn"})
    assert result.status == SUCCESS
    assert result.data["systems"][0]["status"] == "degraded"
    assert executor.check_system_status({"system_name": "mainframe"}).status == NOT_FOUND
