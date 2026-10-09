"""Result types shared by the MCP server (producer) and the agent (consumer)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ToolResult.status values - a small protocol between tools and the LLM.
SUCCESS = "success"
NOT_FOUND = "not_found"
NEEDS_INPUT = "needs_input"
DUPLICATE = "possible_duplicate"
CONFIRMATION_REQUIRED = "confirmation_required"
ERROR = "error"


@dataclass
class Source:
    """Provenance record for data retrieved from a company system (shown in the UI)."""

    kind: str  # kb_article | ticket | employee | system_status
    ref: str
    title: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, "title": self.title}


@dataclass
class ToolResult:
    status: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.status == ERROR

    def to_model_content(self) -> str:
        """Payload sent to the LLM as the tool message (sources are UI-only)."""
        return json.dumps({"status": self.status, "message": self.message, "data": self.data}, default=str)

    def to_wire(self) -> dict[str, Any]:
        """Full payload returned by the MCP server as structured content."""
        return {
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> "ToolResult":
        return cls(
            status=payload.get("status", ERROR),
            message=payload.get("message", ""),
            data=payload.get("data") or {},
            sources=[Source(**s) for s in payload.get("sources") or []],
        )
