"""AgentService - the single entry point the UI (or an API) uses to talk to the agent."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from it_assistant.agent.graph import build_graph
from it_assistant.agent.llm import LLMClient, OpenAILLM
from it_assistant.agent.nodes import AgentNodes
from it_assistant.config import Settings
from it_assistant.tools import definitions as tool_defs
from it_assistant.tools.mcp_backend import MCPToolBackend, ToolBackend
from it_assistant.tools.results import SUCCESS

logger = logging.getLogger(__name__)

# LangGraph recursion limit: each agent step is ~2 graph super-steps, plus entry/exit nodes.
_RECURSION_MARGIN = 8


@dataclass
class TurnResult:
    response: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    employee: dict[str, Any] | None = None
    pending_ticket: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0


class InputValidationError(ValueError):
    pass


class AgentService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        tools: ToolBackend | None = None,
    ) -> None:
        self.settings = settings
        # Tools live in a separate MCP server process; the agent never opens the business database.
        self.tools = tools or MCPToolBackend.from_settings(settings)

        if checkpointer is None:
            settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        self._checkpointer = checkpointer

        self._nodes = AgentNodes(llm or OpenAILLM(settings), self.tools, settings)
        self.graph = build_graph(self._nodes, checkpointer=checkpointer)

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex

    def _validate_input(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise InputValidationError("Please enter a message.")
        text = text.strip()
        if len(text) > self.settings.max_user_message_chars:
            raise InputValidationError(
                f"Message is too long ({len(text)} characters). "
                f"Please keep it under {self.settings.max_user_message_chars} characters."
            )
        return text

    def chat(self, thread_id: str, user_text: str) -> TurnResult:
        text = self._validate_input(user_text)
        started = time.perf_counter()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.max_agent_steps * 2 + _RECURSION_MARGIN,
        }
        turn_input = {
            "user_input": text,
            "workflow_notes": [],
            "confirmation_decision": None,
            "pending_tool_call": None,
            "step_count": 0,
            "trace": [],
            "sources": [],
            "final_response": "",
            "error": None,
        }
        logger.info("thread=%s user_message_chars=%d", thread_id, len(text))
        try:
            final_state = self.graph.invoke(turn_input, config=config)
        except Exception:  # noqa: BLE001 - last line of defence; never show a stack trace to users
            logger.exception("Graph execution failed for thread %s", thread_id)
            return TurnResult(
                response="Something went wrong while processing your request. Please start a new "
                         "conversation, or contact the Service Desk at extension 4357.",
                error="internal_error",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info("thread=%s turn_completed duration_ms=%d steps=%s", thread_id, duration_ms,
                    final_state.get("step_count"))
        return TurnResult(
            response=final_state.get("final_response", ""),
            trace=final_state.get("trace", []),
            sources=_dedupe_sources(final_state.get("sources", [])),
            employee=final_state.get("employee"),
            pending_ticket=final_state.get("pending_ticket"),
            error=final_state.get("error"),
            duration_ms=duration_ms,
        )

    def get_session(self, thread_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        values = snapshot.values or {}
        return {"employee": values.get("employee"), "pending_ticket": values.get("pending_ticket")}

    def system_status(self) -> list[dict[str, Any]]:
        """Current system health for the UI sidebar (read through the MCP server)."""
        result = self.tools.call(tool_defs.CHECK_SYSTEM_STATUS, {})
        return result.data.get("systems", []) if result.status == SUCCESS else []

    def close(self) -> None:
        self.tools.close()

    def reset(self, thread_id: str) -> None:
        try:
            self._checkpointer.delete_thread(thread_id)
        except Exception:  # noqa: BLE001
            logger.exception("Could not delete checkpoint thread %s", thread_id)


def _dedupe_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    seen, unique = set(), []
    for s in sources:
        key = (s["kind"], s["ref"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
