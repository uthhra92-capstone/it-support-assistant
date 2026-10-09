"""Tests for the MCP layer: server contract, client tool policy, and the real stdio transport."""

import json
import typing

from it_assistant.mcp_server import server as mcp_server
from it_assistant.services.validation import TICKET_CATEGORIES, TICKET_PRIORITIES
from it_assistant.tools.mcp_backend import MCPToolBackend
from it_assistant.tools.results import ERROR, NEEDS_INPUT, SUCCESS
from tests.conftest import final, tool_call

THREAD = "mcp-thread"


def _tool(tools, name):
    return next(t["function"] for t in tools.llm_tools() if t["function"]["name"] == name)


# --------------------------------------------------------------- discovery & policy
def test_llm_sees_discovered_tools_but_not_internal_commit(tools):
    names = {t["function"]["name"] for t in tools.llm_tools()}
    assert names == {
        "get_employee_profile", "search_knowledge_base", "lookup_tickets",
        "create_ticket", "check_system_status",
    }
    assert "commit_ticket" not in names


def test_employee_id_is_hidden_from_identity_scoped_tools(tools):
    for name in ("lookup_tickets", "create_ticket"):
        params = _tool(tools, name)["parameters"]
        assert "employee_id" not in params["properties"]
        assert "employee_id" not in params.get("required", [])
    # ...but it is still the input of the verification tool
    assert "employee_id" in _tool(tools, "get_employee_profile")["parameters"]["properties"]


def test_schemas_carry_enums_and_descriptions(tools):
    params = _tool(tools, "create_ticket")["parameters"]
    assert params["properties"]["priority"]["enum"] == list(TICKET_PRIORITIES)
    assert params["properties"]["category"]["enum"] == list(TICKET_CATEGORIES)
    assert "Never claim a ticket was created" in _tool(tools, "create_ticket")["description"]


def test_server_literals_match_validation_rules():
    assert typing.get_args(mcp_server.Category) == TICKET_CATEGORIES
    assert typing.get_args(mcp_server.Priority) == TICKET_PRIORITIES


# ------------------------------------------------------------------- tool calls
def test_call_returns_structured_tool_result(tools):
    result = tools.call("search_knowledge_base", {"query": "reset VPN password"})
    assert result.status == SUCCESS
    assert result.sources[0].ref == "KB-001"


def test_invalid_arguments_become_needs_input(tools):
    result = tools.call("check_system_status", {"system_name": 123})
    assert result.status == NEEDS_INPUT


def test_unknown_tool_on_server_is_error(tools):
    assert tools.call("does_not_exist", {}).status == ERROR


def test_model_cannot_call_internal_commit_tool(service, llm, executor):
    """Even if the model invents a call to commit_ticket, the graph never runs it."""
    llm.add(tool_call("get_employee_profile", {"employee_id": "EMP1024"}), final("Verified."))
    service.chat(THREAD, "EMP1024")
    llm.add(
        tool_call("commit_ticket", {"category": "vpn", "subject": "VPN down", "description": "x" * 20,
                                    "priority": "high", "employee_id": "EMP1024"}),
        final("I can't do that directly."),
    )
    result = service.chat(THREAD, "just create it")
    assert [e["node"] for e in result.trace if e["kind"] == "tool"] == ["unknown_tool"]
    assert executor.tickets.get("INC100246") is None


def test_model_cannot_override_employee_id(service, llm):
    """A forged employee_id argument is replaced with the verified one from state."""
    llm.add(tool_call("get_employee_profile", {"employee_id": "EMP1024"}), final("Verified."))
    service.chat(THREAD, "EMP1024")
    llm.add(tool_call("lookup_tickets", {"employee_id": "EMP1001", "status": "open"}), final("Done."))
    service.chat(THREAD, "show tickets")
    payload = json.loads(llm.requests[-1]["messages"][-1]["content"])
    ids = [t["ticket_id"] for t in payload["data"]["tickets"]]
    assert ids == ["INC100231"]  # EMP1024's ticket, not EMP1001's INC100245


# ---------------------------------------------------------------- real transport
def test_stdio_transport_end_to_end(settings, database):
    """Spawns `python -m it_assistant.mcp_server` and talks to it over stdin/stdout."""
    backend = MCPToolBackend.from_settings(settings)
    try:
        assert len(backend.llm_tools()) == 5
        result = backend.call("lookup_tickets", {"employee_id": "EMP1024", "keyword": "laptop"})
        assert result.status == SUCCESS
        assert result.data["tickets"][0]["ticket_id"] == "INC100231"
    finally:
        backend.close()
