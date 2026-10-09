"""LangGraph state definition.

`llm_messages` uses an append-only reducer: nodes return only the NEW messages and LangGraph
concatenates them. This keeps the chat transcript append-only, so every tool call is always
followed by its matching tool message.

Fields marked "per-turn" are reset by AgentService at the start of every user turn; all other
fields persist across turns via the checkpointer (keyed by thread_id).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class PendingTicket(TypedDict):
    employee_id: str
    draft: dict[str, str]


class AgentState(TypedDict, total=False):
    # ---- persistent conversation memory ----
    llm_messages: Annotated[list[dict[str, Any]], operator.add]
    employee: dict[str, Any] | None          # verified employee profile
    pending_ticket: PendingTicket | None     # draft awaiting explicit user confirmation

    # ---- per-turn ----
    user_input: str
    workflow_notes: list[str]                # application events injected into the next LLM call
    confirmation_decision: str | None        # yes | no | unclear (confirmation gate)
    pending_tool_call: dict[str, Any] | None # the single tool call the LLM requested this step
    step_count: int
    trace: list[dict[str, Any]]              # node/tool timeline shown in the UI
    sources: list[dict[str, str]]            # provenance of retrieved data
    final_response: str
    error: str | None
