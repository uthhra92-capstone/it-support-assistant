"""OpenAI client wrapper (Chat Completions + function calling).

Isolates the OpenAI SDK behind a tiny interface so that:
  * graph nodes deal with plain dicts (checkpoint-serialisable), not SDK objects;
  * API failures are mapped to one exception type with a user-safe message;
  * tests can substitute a scripted fake LLM without network access.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai

from it_assistant.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    message: dict[str, Any]  # assistant message in OpenAI chat format, ready to append to the transcript
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    @property
    def text(self) -> str:
        return (self.message.get("content") or "").strip()

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        calls = []
        for tc in self.message.get("tool_calls") or []:
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                logger.warning("Model produced invalid JSON arguments for %s", tc["function"]["name"])
                args = {}  # the tool then reports the missing fields back to the model
            calls.append({"id": tc["id"], "name": tc["function"]["name"], "input": args})
        return calls


class LLMError(Exception):
    """Raised for any LLM failure; `user_message` is safe to show in the UI."""

    def __init__(self, user_message: str, *, retryable: bool = False) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.retryable = retryable


class LLMClient(Protocol):
    def generate(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


class OpenAILLM:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Credentials come from OPENAI_API_KEY; OPENAI_BASE_URL is honoured if set.
        # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        self._client = openai.OpenAI(
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )

    def generate(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": tools,
            "tool_choice": "auto",
            # One tool per step keeps LangGraph routing deterministic: each step routes to exactly one tool node.
            "parallel_tool_calls": False,
            "max_completion_tokens": self._settings.max_tokens,
        }
        if self._settings.reasoning_effort:
            params["reasoning_effort"] = self._settings.reasoning_effort

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**params)
        except openai.AuthenticationError as exc:
            logger.error("OpenAI authentication failed: %s", exc)
            raise LLMError("The AI service rejected our credentials. Please check OPENAI_API_KEY.") from exc
        except openai.PermissionDeniedError as exc:
            logger.error("OpenAI permission denied: %s", exc)
            raise LLMError("The AI service denied access for this API key.") from exc
        except openai.NotFoundError as exc:
            logger.error("Model or endpoint not found (%s): %s", self._settings.model, exc)
            raise LLMError("The configured AI model is unavailable. Please check OPENAI_MODEL.") from exc
        except openai.BadRequestError as exc:
            logger.error("Bad request to OpenAI API: %s", exc)
            raise LLMError("The AI service could not process this request. Please start a new conversation.") from exc
        except openai.RateLimitError as exc:
            logger.warning("Rate limited / quota exceeded on OpenAI API: %s", exc)
            raise LLMError("The AI service is busy or the quota is exhausted. Please try again shortly.",
                           retryable=True) from exc
        except openai.APITimeoutError as exc:
            logger.warning("OpenAI API timeout: %s", exc)
            raise LLMError("The AI service took too long to respond. Please try again.", retryable=True) from exc
        except openai.APIConnectionError as exc:
            logger.warning("Cannot reach OpenAI API: %s", exc)
            raise LLMError("Cannot reach the AI service. Check the network connection and try again.",
                           retryable=True) from exc
        except openai.APIStatusError as exc:
            logger.error("OpenAI API error %s: %s", exc.status_code, exc)
            raise LLMError("The AI service returned an error. Please try again shortly.", retryable=True) from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        choice = completion.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason
        if getattr(msg, "refusal", None):
            finish_reason = "refusal"

        message: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            # Only function tool calls are defined; keep the first one (parallel calls are disabled,
            # and every tool_call id in the transcript must be answered by a tool message).
            first = msg.tool_calls[0]
            message["tool_calls"] = [{
                "id": first.id,
                "type": "function",
                "function": {"name": first.function.name, "arguments": first.function.arguments},
            }]

        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else None,
            "output_tokens": completion.usage.completion_tokens if completion.usage else None,
            "latency_ms": elapsed_ms,
        }
        request_id = getattr(completion, "_request_id", None)
        logger.info(
            "LLM call model=%s finish=%s in=%s out=%s latency_ms=%s request_id=%s",
            completion.model, finish_reason, usage["input_tokens"], usage["output_tokens"],
            elapsed_ms, request_id,
        )
        return LLMResponse(message=message, finish_reason=finish_reason, usage=usage, request_id=request_id)
