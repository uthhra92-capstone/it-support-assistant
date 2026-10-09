"""Streamlit UI for the AI IT Support Assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import atexit
import logging

import streamlit as st

from it_assistant.agent.service import AgentService, InputValidationError
from it_assistant.config import get_settings, has_llm_credentials
from it_assistant.logging_config import setup_logging
from it_assistant.ui import components

settings = get_settings()
setup_logging(settings)
logger = logging.getLogger("it_assistant.ui")

st.set_page_config(page_title=settings.app_title, page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="Starting the assistant and its MCP tool server…")
def get_service() -> AgentService:
    service = AgentService(settings)
    atexit.register(service.close)  # stop the MCP server subprocess on shutdown
    return service


def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = AgentService.new_thread_id()
        st.session_state.messages = []
        st.session_state.queued_prompt = None
        logger.info("New UI session thread=%s", st.session_state.thread_id)


def reset_conversation(service: AgentService) -> None:
    service.reset(st.session_state.thread_id)
    for key in ("thread_id", "messages", "queued_prompt"):
        st.session_state.pop(key, None)
    init_session()


def handle_prompt(service: AgentService, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🛠️"):
        with st.spinner("Working on it…"):
            try:
                result = service.chat(st.session_state.thread_id, prompt)
            except InputValidationError as exc:
                st.session_state.messages.pop()
                st.warning(str(exc))
                return
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.response,
        "trace": result.trace,
        "sources": result.sources,
        "error": result.error,
        "duration_ms": result.duration_ms,
    })
    st.rerun()


def main() -> None:
    if not has_llm_credentials():
        st.error(
            "**OPENAI_API_KEY is not set.** Copy `.env.example` to `.env`, add your key, "
            "and restart the app."
        )
        st.stop()

    try:
        service = get_service()
    except Exception as exc:  # noqa: BLE001 - show a clean startup error instead of a traceback
        logger.exception("Failed to start AgentService")
        st.error(f"The assistant could not start: {exc}")
        st.stop()

    init_session()
    session = service.get_session(st.session_state.thread_id)

    with st.sidebar:
        if components.render_sidebar(settings, service, session):
            reset_conversation(service)
            st.rerun()

    st.title(f"🛠️ {settings.app_title}")
    st.caption(
        f"Ask about IT how-tos, your existing tickets, system status, or raise a new ticket - "
        f"{settings.organization_name} IT Service Desk."
    )

    if not st.session_state.messages:
        components.render_welcome()

    for message in st.session_state.messages:
        components.render_message(message)

    if session.get("pending_ticket"):
        decision = components.render_pending_ticket(session["pending_ticket"])
        if decision:
            st.session_state.queued_prompt = decision

    prompt = st.chat_input("Describe your IT issue or ask a question…", max_chars=settings.max_user_message_chars)
    queued = st.session_state.pop("queued_prompt", None)
    st.session_state.queued_prompt = None
    if prompt or queued:
        handle_prompt(service, prompt or queued)


main()
