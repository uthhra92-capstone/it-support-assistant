"""Tool (function) schemas exposed to the LLM via OpenAI function calling.

Note that ticket tools deliberately have NO employee_id parameter: the employee's
identity comes from verified graph state, never from model-generated arguments.
"""

from __future__ import annotations

from it_assistant.services.validation import (
    TICKET_CATEGORIES,
    TICKET_PRIORITIES,
    TICKET_STATUS_FILTERS,
)

GET_EMPLOYEE_PROFILE = "get_employee_profile"
SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
LOOKUP_TICKETS = "lookup_tickets"
CREATE_TICKET = "create_ticket"
CHECK_SYSTEM_STATUS = "check_system_status"

_TOOLS: list[dict] = [
    {
        "name": GET_EMPLOYEE_PROFILE,
        "description": (
            "Verify an employee and load their profile (name, department, location, assigned "
            "devices). Call this as soon as the user provides an employee ID (format EMP followed "
            "by 4 digits, e.g. EMP1024). A successful call marks the employee as verified for the "
            "rest of the conversation; ticket lookup and ticket creation require it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "Employee ID exactly as provided by the user, e.g. EMP1024.",
                }
            },
            "required": ["employee_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": SEARCH_KNOWLEDGE_BASE,
        "description": (
            "Search the internal IT knowledge base for how-to guides and troubleshooting steps "
            "(VPN, passwords, MFA, email, printers, Wi-Fi, software installs, laptops, etc.). "
            "Use for any 'how do I...' or troubleshooting question. Returns the best matching "
            "articles with their IDs; answer only from returned article content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short keyword-rich search query, e.g. 'reset VPN password'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": LOOKUP_TICKETS,
        "description": (
            "Look up the verified employee's existing support tickets. Use for questions like "
            "'what is the status of my laptop issue' or 'do I have open tickets'. Requires a "
            "verified employee (call get_employee_profile first). All filters are optional."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "Specific ticket ID if the user gave one, format INC followed by 6 digits.",
                },
                "status": {
                    "type": "string",
                    "enum": list(TICKET_STATUS_FILTERS),
                    "description": "Filter by lifecycle: open (incl. in progress/on hold), closed, or all. Default all.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Single keyword to match subject/description/category, e.g. 'laptop' or 'vpn'.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": CREATE_TICKET,
        "description": (
            "Prepare a new IT support ticket for the verified employee. Only call this once you "
            "have gathered a clear subject and a description with enough detail (what is "
            "happening, since when, error messages). The application validates the request, "
            "checks for duplicate open tickets and then asks the user to confirm - the ticket is "
            "NOT created until the user confirms. Never claim a ticket was created based on this "
            "tool's result alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": list(TICKET_CATEGORIES)},
                "subject": {
                    "type": "string",
                    "description": "One-line summary, 5-120 characters.",
                },
                "description": {
                    "type": "string",
                    "description": "Details of the problem in the user's own terms, at least 15 characters.",
                },
                "priority": {
                    "type": "string",
                    "enum": list(TICKET_PRIORITIES),
                    "description": (
                        "low: minor inconvenience; medium: work impacted but workaround exists; "
                        "high: cannot work; critical: many users or a security incident."
                    ),
                },
                "confirmed_not_duplicate": {
                    "type": "boolean",
                    "description": (
                        "Set true ONLY if a previous call reported a possible duplicate and the user "
                        "explicitly said this is a different issue."
                    ),
                },
            },
            "required": ["category", "subject", "description", "priority"],
            "additionalProperties": False,
        },
    },
    {
        "name": CHECK_SYSTEM_STATUS,
        "description": (
            "Check the current health of company IT systems (VPN, Email, Wi-Fi, SSO, HR Portal, "
            "etc.). Use when the user reports something is down or slow, before suggesting "
            "troubleshooting or raising a ticket - a known outage may already explain it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "system_name": {
                    "type": "string",
                    "description": "Optional system name, e.g. 'VPN'. Omit to list all systems.",
                }
            },
            "additionalProperties": False,
        },
    },
]

# OpenAI Chat Completions function-calling format.
TOOL_DEFINITIONS: list[dict] = [{"type": "function", "function": tool} for tool in _TOOLS]
