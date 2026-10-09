"""Shared fixtures: temp database, a real in-process MCP tool server, and a scripted fake LLM."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from it_assistant.agent.llm import LLMError, LLMResponse
from it_assistant.agent.service import AgentService
from it_assistant.config import PROJECT_ROOT, Settings
from it_assistant.data.database import Database
from it_assistant.mcp_server.server import create_server
from it_assistant.tools.mcp_backend import MCPToolBackend
from it_assistant.tools.executor import ToolExecutor

_ids = itertools.count(1)


def tool_call(name: str, args: dict[str, Any] | None = None) -> LLMResponse:
    return LLMResponse(
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{next(_ids)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args or {})},
            }],
        },
        finish_reason="tool_calls",
    )


def final(text: str) -> LLMResponse:
    return LLMResponse(message={"role": "assistant", "content": text}, finish_reason="stop")


class ScriptedLLM:
    """Returns pre-defined responses in order and records every request it receives."""

    def __init__(self, responses: Iterable[LLMResponse | Exception] = ()) -> None:
        self.queue = list(responses)
        self.requests: list[dict[str, Any]] = []

    def add(self, *responses: LLMResponse | Exception) -> None:
        self.queue.extend(responses)

    def generate(self, *, system, messages, tools) -> LLMResponse:
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.queue:
            raise AssertionError("ScriptedLLM ran out of responses")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        database_path=tmp_path / "test.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        seed_dir=PROJECT_ROOT / "data" / "seed",
        log_dir=tmp_path / "logs",
        max_agent_steps=6,
    )


@pytest.fixture
def database(settings: Settings) -> Database:
    db = Database(settings.database_path)
    db.initialize(settings.seed_dir)
    return db


@pytest.fixture
def executor(database: Database, settings: Settings) -> ToolExecutor:
    return ToolExecutor(database, settings)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def tools(settings: Settings, database: Database):
    """The real MCP server, connected in-process (same protocol, no subprocess)."""
    backend = MCPToolBackend(create_server(settings), call_timeout=10, startup_timeout=10)
    yield backend
    backend.close()


@pytest.fixture
def service(settings: Settings, llm: ScriptedLLM, tools: MCPToolBackend) -> AgentService:
    return AgentService(settings, llm=llm, checkpointer=InMemorySaver(), tools=tools)


__all__ = ["ScriptedLLM", "tool_call", "final", "LLMError"]
