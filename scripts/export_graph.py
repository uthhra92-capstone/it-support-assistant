"""Export the compiled LangGraph workflow as a Mermaid diagram (docs/graph.mmd).

The diagram is generated from the real graph object, so it can never drift from the code.

Usage:
    python -m scripts.export_graph
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from langgraph.checkpoint.memory import InMemorySaver

from it_assistant.agent.service import AgentService
from it_assistant.config import PROJECT_ROOT, get_settings
from it_assistant.mcp_server.server import create_server
from it_assistant.tools.mcp_backend import MCPToolBackend


class _NoLLM:
    def generate(self, **_kwargs):  # pragma: no cover - never called
        raise RuntimeError("not used")


def main() -> None:
    with TemporaryDirectory() as tmp:
        settings = replace(get_settings(), database_path=Path(tmp) / "graph.db")
        tools = MCPToolBackend(create_server(settings), call_timeout=10, startup_timeout=10)
        try:
            service = AgentService(settings, llm=_NoLLM(), checkpointer=InMemorySaver(), tools=tools)
            mermaid = service.graph.get_graph().draw_mermaid()
        finally:
            tools.close()
    out = PROJECT_ROOT / "docs" / "graph.mmd"
    out.parent.mkdir(exist_ok=True)
    out.write_text(mermaid, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
