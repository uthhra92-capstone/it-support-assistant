"""Application configuration loaded from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_REASONING_EFFORTS = {"", "minimal", "low", "medium", "high"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw) if raw and raw.strip() else default
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    # LLM
    model: str
    reasoning_effort: str  # "" = don't send (required for non-reasoning models such as gpt-4.1)
    max_tokens: int
    request_timeout_seconds: float
    max_retries: int

    # MCP tool server
    mcp_call_timeout_seconds: float
    mcp_startup_timeout_seconds: float

    # Agent behaviour
    max_agent_steps: int
    max_user_message_chars: int
    kb_top_k: int
    kb_min_score: float
    duplicate_similarity_threshold: float

    # Storage
    database_path: Path
    checkpoint_db_path: Path
    seed_dir: Path

    # Observability
    log_level: str
    log_dir: Path

    # UI
    app_title: str
    organization_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env", override=False)

        reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()
        if reasoning_effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"OPENAI_REASONING_EFFORT must be empty or one of "
                f"{sorted(VALID_REASONING_EFFORTS - {''})}, got '{reasoning_effort}'"
            )

        settings = cls(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            reasoning_effort=reasoning_effort,
            max_tokens=_env_int("LLM_MAX_TOKENS", 8000),
            request_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 90.0),
            max_retries=_env_int("LLM_MAX_RETRIES", 3),
            mcp_call_timeout_seconds=_env_float("MCP_CALL_TIMEOUT_SECONDS", 30.0),
            mcp_startup_timeout_seconds=_env_float("MCP_STARTUP_TIMEOUT_SECONDS", 30.0),
            max_agent_steps=_env_int("MAX_AGENT_STEPS", 6),
            max_user_message_chars=_env_int("MAX_USER_MESSAGE_CHARS", 2000),
            kb_top_k=_env_int("KB_TOP_K", 3),
            kb_min_score=_env_float("KB_MIN_SCORE", 0.5),
            duplicate_similarity_threshold=_env_float("DUPLICATE_SIMILARITY_THRESHOLD", 0.35),
            database_path=_env_path("DATABASE_PATH", Path("var/it_support.db")),
            checkpoint_db_path=_env_path("CHECKPOINT_DB_PATH", Path("var/checkpoints.db")),
            seed_dir=_env_path("SEED_DATA_DIR", Path("data/seed")),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            log_dir=_env_path("LOG_DIR", Path("var/logs")),
            app_title=os.getenv("APP_TITLE", "AI IT Support Assistant"),
            organization_name=os.getenv("ORGANIZATION_NAME", "Contoso Ltd."),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.max_agent_steps < 1:
            raise ValueError("MAX_AGENT_STEPS must be >= 1")
        if self.max_tokens < 256:
            raise ValueError("LLM_MAX_TOKENS must be >= 256")
        if not 0 < self.duplicate_similarity_threshold <= 1:
            raise ValueError("DUPLICATE_SIMILARITY_THRESHOLD must be in (0, 1]")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def has_llm_credentials() -> bool:
    """True when the OpenAI SDK can find an API key in the environment."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return bool(os.getenv("OPENAI_API_KEY"))
