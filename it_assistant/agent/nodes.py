"""LangGraph node implementations and routing functions."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from it_assistant.agent.llm import LLMClient, LLMError
from it_assistant.agent.prompts import build_session_context, build_system_prompt
from it_assistant.agent.state import AgentState
from it_assistant.config import Settings
from it_assistant.services.validation import TICKET_ID_IN_TEXT_RE
from it_assistant.tools import definitions as tool_defs
from it_assistant.tools.executor import CONFIRMATION_REQUIRED, SUCCESS, ToolExecutor, ToolResult

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- node names
INGEST = "ingest_user_message"
CONFIRMATION_GATE = "confirmation_gate"
COMMIT_TICKET = "commit_ticket"
AGENT = "agent"
EMPLOYEE_LOOKUP = "employee_lookup"
KNOWLEDGE_SEARCH = "knowledge_search"
TICKET_LOOKUP = "ticket_lookup"
TICKET_CREATION = "ticket_creation"
SYSTEM_STATUS = "system_status"
UNKNOWN_TOOL = "unknown_tool"
RESPOND = "respond"

TOOL_NODE_BY_NAME = {
    tool_defs.GET_EMPLOYEE_PROFILE: EMPLOYEE_LOOKUP,
    tool_defs.SEARCH_KNOWLEDGE_BASE: KNOWLEDGE_SEARCH,
    tool_defs.LOOKUP_TICKETS: TICKET_LOOKUP,
    tool_defs.CREATE_TICKET: TICKET_CREATION,
    tool_defs.CHECK_SYSTEM_STATUS: SYSTEM_STATUS,
}

FALLBACK_REPLY = (
    "Sorry, I couldn't complete that request. Please try rephrasing, or contact the "
    "Service Desk at extension 4357."
)

# ----------------------------------------------------------------- confirmation classifier
_YES = {
    "y", "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "correct",
    "go ahead", "please do", "do it", "create it", "raise it", "submit", "submit it", "proceed",
    "yes please", "yes create it", "yes go ahead", "looks good", "that's right", "thats right",
}
_NO = {
    "n", "no", "nope", "nah", "cancel", "stop", "don't", "dont", "do not", "never mind",
    "nevermind", "no thanks", "no thank you", "cancel it", "discard", "abort",
}


def classify_confirmation(text: str) -> str:
    """Deterministically classify a reply to 'shall I create this ticket?' as yes / no / unclear.

    Anything that is not a short, unambiguous yes or no (e.g. "yes but make it high priority")
    is 'unclear' and handed to the LLM, which can revise the draft.
    """
    normalized = re.sub(r"[^\w\s']", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\b(please|thanks|thank you)$", "", normalized).strip()
    if normalized in _YES:
        return "yes"
    if normalized in _NO:
        return "no"
    return "unclear"


def sanitize_user_text(text: str) -> str:
    """Stop users from forging application events inside their own message."""
    return re.sub(r"<\s*/?\s*workflow_event", "&lt;workflow_event", text, flags=re.IGNORECASE)


def _trace(node: str, kind: str, label: str, started: float, **detail: Any) -> dict[str, Any]:
    return {
        "node": node,
        "kind": kind,
        "label": label,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "detail": detail,
    }


class AgentNodes:
    def __init__(self, llm: LLMClient, executor: ToolExecutor, settings: Settings) -> None:
        self._llm = llm
        self._executor = executor
        self._settings = settings
        self._system_prompt = build_system_prompt(settings.organization_name)

    # ============================================================== entry / HITL nodes
    def confirmation_gate(self, state: AgentState) -> dict[str, Any]:
        """Human-in-the-loop checkpoint: runs only when a ticket draft is awaiting confirmation."""
        started = time.perf_counter()
        decision = classify_confirmation(state.get("user_input", ""))
        pending = state["pending_ticket"]
        update: dict[str, Any] = {"trace": state.get("trace", [])}
        notes = list(state.get("workflow_notes", []))

        if decision == "no":
            notes.append("The user declined the pending ticket draft. It was discarded; no ticket was created.")
            update["pending_ticket"] = None
        elif decision == "unclear":
            notes.append(
                "A ticket draft is still awaiting confirmation and the user's reply was not a clear "
                "yes or no. If they asked for changes, call create_ticket again with the updated "
                "details; otherwise answer them and ask again whether to create the ticket."
            )
        update["workflow_notes"] = notes
        update["trace"] = update["trace"] + [
            _trace(CONFIRMATION_GATE, "guardrail", f"User confirmation: {decision}", started,
                   draft=pending["draft"])
        ]
        update["confirmation_decision"] = decision  # consumed by route_after_confirmation
        return update

    def commit_ticket(self, state: AgentState) -> dict[str, Any]:
        """Writes the confirmed ticket to the database (the only node that does)."""
        started = time.perf_counter()
        pending = state["pending_ticket"]
        result = self._executor.commit_ticket(pending["draft"], pending["employee_id"])
        notes = list(state.get("workflow_notes", []))
        if result.status == SUCCESS:
            notes.append(
                "The user confirmed the draft and the ticket was created in the ticketing system: "
                + json.dumps(result.data["ticket"])
            )
        else:
            notes.append(f"The user confirmed, but ticket creation FAILED: {result.message}")
        return {
            "pending_ticket": None if result.status == SUCCESS else pending,
            "workflow_notes": notes,
            "sources": state.get("sources", []) + [s.to_dict() for s in result.sources],
            "trace": state.get("trace", []) + [
                _trace(COMMIT_TICKET, "tool", "Create ticket (confirmed)", started,
                       status=result.status, result=result.data or result.message)
            ],
        }

    def ingest_user_message(self, state: AgentState) -> dict[str, Any]:
        """Turns the raw user input (plus any application events) into a chat user message."""
        parts = [sanitize_user_text(state["user_input"])]
        for note in state.get("workflow_notes", []):
            parts.append(f'<workflow_event source="application">{note}</workflow_event>')
        return {"llm_messages": [{"role": "user", "content": "\n\n".join(parts)}], "workflow_notes": []}

    # ======================================================================= agent node
    def agent(self, state: AgentState) -> dict[str, Any]:
        """LLM decision node: the model either requests one tool or writes the final answer."""
        started = time.perf_counter()
        step = state.get("step_count", 0) + 1
        trace = state.get("trace", [])

        if step > self._settings.max_agent_steps:
            logger.warning("Max agent steps (%s) reached", self._settings.max_agent_steps)
            text = (
                "I wasn't able to finish this in a reasonable number of steps. Here's what I have "
                "so far - please rephrase or narrow down the request, or contact the Service Desk "
                "at extension 4357."
            )
            return {
                "step_count": step,
                "pending_tool_call": None,
                "final_response": text,
                "llm_messages": [{"role": "assistant", "content": text}],
                "trace": trace + [_trace(AGENT, "guardrail", "Step limit reached", started)],
            }

        system = (
            self._system_prompt + "\n\n"
            + build_session_context(state.get("employee"), state.get("pending_ticket"))
        )
        try:
            response = self._llm.generate(
                system=system, messages=state["llm_messages"], tools=tool_defs.TOOL_DEFINITIONS
            )
        except LLMError as exc:
            # Keep the transcript well-formed (user/assistant alternation) for the next turn.
            return {
                "step_count": step,
                "pending_tool_call": None,
                "error": exc.user_message,
                "final_response": exc.user_message,
                "llm_messages": [{"role": "assistant",
                                  "content": "(The previous request failed due to a temporary service error.)"}],
                "trace": trace + [_trace(AGENT, "error", "LLM call failed", started, error=exc.user_message)],
            }

        tool_calls = response.tool_calls
        if response.finish_reason == "tool_calls" and tool_calls:
            call = tool_calls[0]
            return {
                "step_count": step,
                "pending_tool_call": call,
                "llm_messages": [response.message],
                "trace": trace + [_trace(AGENT, "decision", f"Decided to call {call['name']}", started,
                                         tool=call["name"], arguments=call["input"], usage=response.usage)],
            }

        if response.finish_reason in {"refusal", "content_filter"}:
            text = "I'm not able to help with that request. For IT issues, I'm happy to help."
        else:
            text = response.text or FALLBACK_REPLY
        # Store a plain text message: an unanswered tool call (e.g. after finish_reason="length")
        # would make the next request invalid.
        return {
            "step_count": step,
            "pending_tool_call": None,
            "final_response": text,
            "llm_messages": [{"role": "assistant", "content": text}],
            "trace": trace + [_trace(AGENT, "decision", "Composed final answer (no tool needed)"
                                     if step == 1 else "Composed final answer from tool results",
                                     started, finish_reason=response.finish_reason, usage=response.usage)],
        }

    # ======================================================================= tool nodes
    def _run_tool(
        self, state: AgentState, node: str, label: str, fn: Callable[[dict[str, Any]], ToolResult],
        extra_update: Callable[[ToolResult, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        call = state["pending_tool_call"]
        args = call.get("input") or {}
        try:
            result = fn(args)
        except Exception:  # noqa: BLE001 - a tool must never crash the conversation
            logger.exception("Tool %s crashed", call["name"])
            result = ToolResult("error", "The tool failed unexpectedly. Tell the user and suggest trying again later.")
        logger.info("Tool %s status=%s", call["name"], result.status)

        # Errors are signalled to the model through the "status" field of the JSON payload.
        tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result.to_model_content()}

        update: dict[str, Any] = {
            "pending_tool_call": None,
            "llm_messages": [tool_message],
            "sources": state.get("sources", []) + [s.to_dict() for s in result.sources],
            "trace": state.get("trace", []) + [
                _trace(node, "tool", label, started, tool=call["name"], arguments=args,
                       status=result.status, message=result.message, result=result.data)
            ],
        }
        if extra_update:
            update.update(extra_update(result, args))
        return update

    def employee_lookup(self, state: AgentState) -> dict[str, Any]:
        def remember_employee(result: ToolResult, _args: dict[str, Any]) -> dict[str, Any]:
            if result.status != SUCCESS:
                return {}
            employee = result.data["employee"]
            previous = state.get("employee")
            update: dict[str, Any] = {"employee": employee}
            if previous and previous["employee_id"] != employee["employee_id"]:
                update["pending_ticket"] = None  # identity changed - drop the other person's draft
            return update

        return self._run_tool(state, EMPLOYEE_LOOKUP, "Employee lookup",
                              self._executor.get_employee_profile, remember_employee)

    def knowledge_search(self, state: AgentState) -> dict[str, Any]:
        return self._run_tool(state, KNOWLEDGE_SEARCH, "Knowledge base search",
                              self._executor.search_knowledge_base)

    def ticket_lookup(self, state: AgentState) -> dict[str, Any]:
        employee_id = (state.get("employee") or {}).get("employee_id")
        return self._run_tool(state, TICKET_LOOKUP, "Ticket lookup",
                              lambda args: self._executor.lookup_tickets(args, employee_id))

    def ticket_creation(self, state: AgentState) -> dict[str, Any]:
        """Validates and duplicate-checks the request, then stages it for user confirmation."""
        employee_id = (state.get("employee") or {}).get("employee_id")

        def stage_draft(result: ToolResult, _args: dict[str, Any]) -> dict[str, Any]:
            if result.status == CONFIRMATION_REQUIRED and employee_id:
                return {"pending_ticket": {"employee_id": employee_id, "draft": result.data["draft"]}}
            return {"pending_ticket": None}

        return self._run_tool(state, TICKET_CREATION, "Ticket creation - validate & stage",
                              lambda args: self._executor.prepare_ticket(args, employee_id), stage_draft)

    def system_status(self, state: AgentState) -> dict[str, Any]:
        return self._run_tool(state, SYSTEM_STATUS, "System status check",
                              self._executor.check_system_status)

    def unknown_tool(self, state: AgentState) -> dict[str, Any]:
        name = state["pending_tool_call"]["name"]
        return self._run_tool(state, UNKNOWN_TOOL, f"Unknown tool '{name}'",
                              lambda _args: ToolResult("error", f"Tool '{name}' does not exist."))

    # ===================================================================== respond node
    def respond(self, state: AgentState) -> dict[str, Any]:
        """Final guardrail: verify every ticket ID mentioned in the answer really exists."""
        started = time.perf_counter()
        text = state.get("final_response") or FALLBACK_REPLY
        employee_id = (state.get("employee") or {}).get("employee_id")

        unverified = []
        for ticket_id in sorted(set(TICKET_ID_IN_TEXT_RE.findall(text))):
            ticket = self._executor.tickets.get(ticket_id)
            if ticket is None or ticket.employee_id != employee_id:
                unverified.append(ticket_id)
        trace = state.get("trace", [])
        if unverified:
            logger.warning("Response referenced unverified ticket IDs: %s", unverified)
            text += (
                "\n\n> ⚠️ Note: I could not verify " + ", ".join(unverified)
                + " in the ticketing system. Please treat that reference as unconfirmed."
            )
        trace = trace + [_trace(RESPOND, "guardrail", "Response validated", started,
                                unverified_ticket_ids=unverified)]
        return {"final_response": text, "trace": trace}


# =============================================================================== routing
def route_entry(state: AgentState) -> str:
    return CONFIRMATION_GATE if state.get("pending_ticket") else INGEST


def route_after_confirmation(state: AgentState) -> str:
    return COMMIT_TICKET if state.get("confirmation_decision") == "yes" else INGEST


def route_after_agent(state: AgentState) -> str:
    call = state.get("pending_tool_call")
    if not call:
        return RESPOND
    return TOOL_NODE_BY_NAME.get(call["name"], UNKNOWN_TOOL)
