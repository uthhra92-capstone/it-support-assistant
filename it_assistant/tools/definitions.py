"""Tool names and the agent-side tool policy.

The tool schemas themselves are owned by the MCP server (it_assistant/mcp_server/server.py)
and discovered at runtime with `list_tools`. This module decides what the LLM may see:

* INTERNAL_TOOLS are never offered to the LLM. `commit_ticket` writes to the database, so it
  can only be reached through the human-in-the-loop confirmation gate in the graph.
* STATE_INJECTED_ARGS are removed from the schema shown to the LLM and filled in from verified
  graph state on every call. The model therefore cannot act on behalf of another employee.
"""

from __future__ import annotations

GET_EMPLOYEE_PROFILE = "get_employee_profile"
SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
LOOKUP_TICKETS = "lookup_tickets"
CREATE_TICKET = "create_ticket"
COMMIT_TICKET = "commit_ticket"
CHECK_SYSTEM_STATUS = "check_system_status"

INTERNAL_TOOLS = frozenset({COMMIT_TICKET})

STATE_INJECTED_ARGS: dict[str, frozenset[str]] = {
    LOOKUP_TICKETS: frozenset({"employee_id"}),
    CREATE_TICKET: frozenset({"employee_id"}),
    COMMIT_TICKET: frozenset({"employee_id"}),
}
