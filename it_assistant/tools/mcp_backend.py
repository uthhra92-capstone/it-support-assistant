"""MCP client side: connects the agent to the IT tools MCP server.

LangGraph nodes and Streamlit are synchronous, while the MCP SDK is async. `MCPToolBackend`
runs the MCP client on a dedicated event-loop thread and exposes a small blocking API:

    backend.llm_tools()           -> tool schemas in OpenAI function-calling format
    backend.call(name, args)      -> ToolResult (never raises)
"""

from __future__ import annotations

import asyncio
import copy
import logging
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Protocol

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from it_assistant.config import PROJECT_ROOT, Settings
from it_assistant.tools.definitions import INTERNAL_TOOLS, STATE_INJECTED_ARGS
from it_assistant.tools.results import ERROR, NEEDS_INPUT, ToolResult

logger = logging.getLogger(__name__)


class ToolBackend(Protocol):
    def llm_tools(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, args: dict[str, Any]) -> ToolResult: ...

    def close(self) -> None: ...


class MCPConnectionError(RuntimeError):
    pass


def _strip_titles(schema: Any) -> Any:
    """Remove pydantic's auto-generated 'title' keys - they add tokens and no meaning."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI function tools, applying the tool policy."""
    tools = []
    for tool in mcp_tools:
        if tool.name in INTERNAL_TOOLS:
            continue
        schema = _strip_titles(copy.deepcopy(tool.input_schema))
        hidden = STATE_INJECTED_ARGS.get(tool.name, frozenset())
        schema["properties"] = {k: v for k, v in schema.get("properties", {}).items() if k not in hidden}
        schema["required"] = [r for r in schema.get("required", []) if r not in hidden]
        schema["additionalProperties"] = False
        tools.append({
            "type": "function",
            "function": {"name": tool.name, "description": tool.description or "", "parameters": schema},
        })
    return tools


class MCPToolBackend:
    def __init__(self, server: Any, *, call_timeout: float, startup_timeout: float) -> None:
        """`server` is anything mcp.Client accepts: StdioServerParameters (production) or an
        in-process MCPServer instance (tests)."""
        self._server = server
        self._call_timeout = call_timeout
        self._client: Client | None = None
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        self._stop = asyncio.Event()
        self._owner = asyncio.run_coroutine_threadsafe(self._own_connection(), self._loop)

        if not self._ready.wait(startup_timeout) or self._client is None:
            self._shutdown_loop()
            raise MCPConnectionError(f"Could not start the MCP tool server: {self._startup_error or 'timed out'}")
        self._tools = self._list_tools()
        logger.info("Connected to MCP server; tools offered to LLM: %s",
                    [t["function"]["name"] for t in self._tools])

    @classmethod
    def from_settings(cls, settings: Settings) -> "MCPToolBackend":
        """Launch `python -m it_assistant.mcp_server` as a subprocess and connect over stdio."""
        env = {
            "DATABASE_PATH": str(settings.database_path),
            "SEED_DATA_DIR": str(settings.seed_dir),
            "LOG_DIR": str(settings.log_dir),
            "LOG_LEVEL": settings.log_level,
            "KB_TOP_K": str(settings.kb_top_k),
            "KB_MIN_SCORE": str(settings.kb_min_score),
            "DUPLICATE_SIMILARITY_THRESHOLD": str(settings.duplicate_similarity_threshold),
            "PYTHONIOENCODING": "utf-8",
        }
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "it_assistant.mcp_server"], env=env, cwd=str(PROJECT_ROOT)
        )
        return cls(params, call_timeout=settings.mcp_call_timeout_seconds,
                   startup_timeout=settings.mcp_startup_timeout_seconds)

    # ------------------------------------------------------------------ lifecycle
    async def _own_connection(self) -> None:
        # The connection is opened and closed inside ONE task, as the MCP SDK (anyio) requires.
        try:
            async with Client(self._server, read_timeout_seconds=self._call_timeout) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - reported to the constructor / logs
            self._startup_error = exc
            logger.exception("MCP connection failed or closed unexpectedly")
        finally:
            self._client = None
            self._ready.set()

    def _run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self._call_timeout + 5)

    def _list_tools(self) -> list[dict[str, Any]]:
        assert self._client is not None
        result = self._run(self._client.list_tools())
        return to_openai_tools(result.tools)

    def close(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._owner.result(timeout=10)
        except Exception:  # noqa: BLE001
            logger.warning("MCP connection did not close cleanly")
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()

    # ---------------------------------------------------------------------- API
    def llm_tools(self) -> list[dict[str, Any]]:
        return self._tools

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        client = self._client
        if client is None:
            return ToolResult(ERROR, "The IT tools service is unavailable. Tell the user and suggest trying again later.")
        try:
            result = self._run(client.call_tool(name, args))
        except (FutureTimeout, TimeoutError):
            logger.error("MCP tool %s timed out", name)
            return ToolResult(ERROR, "The IT tools service timed out. Tell the user and suggest trying again later.")
        except Exception:  # noqa: BLE001 - transport failure must not crash the conversation
            logger.exception("MCP call to %s failed", name)
            return ToolResult(ERROR, "The IT tools service is unavailable. Tell the user and suggest trying again later.")

        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content).strip()
            logger.warning("MCP tool %s returned an error: %s", name, text[:300])
            if "validation error" in text.lower():
                # Bad arguments from the model: give it the details so it can correct itself.
                return ToolResult(NEEDS_INPUT, f"Invalid tool arguments: {text[:500]}")
            return ToolResult(ERROR, "The tool failed unexpectedly. Tell the user and suggest trying again later.")
        if not isinstance(result.structured_content, dict):
            logger.error("MCP tool %s returned no structured content", name)
            return ToolResult(ERROR, "The tool returned an unexpected response.")
        return ToolResult.from_wire(result.structured_content)
