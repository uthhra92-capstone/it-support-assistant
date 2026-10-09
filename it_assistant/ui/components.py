"""Reusable Streamlit rendering helpers (kept out of app.py to keep it readable)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from it_assistant.agent.service import AgentService
from it_assistant.config import Settings

SAMPLE_PROMPTS = [
    "How do I reset my VPN password?",
    "What is the status of my laptop issue?",
    "My VPN is not working. Please raise a ticket.",
    "Is email down right now?",
    "How do I move my MFA to a new phone?",
]

DEMO_EMPLOYEES = {
    "EMP1024": "Ananya Rao - has a laptop ticket in progress",
    "EMP1001": "Priya Sharma - has an open VPN ticket (duplicate demo)",
    "EMP1007": "Grace Kim - Wi-Fi ticket",
    "EMP1006": "Tom Becker - inactive account (validation demo)",
}

KIND_ICON = {"tool": "🔧", "decision": "🧠", "guardrail": "🛡️", "error": "❌"}
SOURCE_ICON = {"kb_article": "📘", "ticket": "🎫", "employee": "👤", "system_status": "📡"}
STATUS_ICON = {"operational": "🟢", "degraded": "🟠", "outage": "🔴", "maintenance": "🔵"}


def render_sidebar(settings: Settings, service: AgentService, session: dict[str, Any]) -> bool:
    """Draws the sidebar. Returns True if the user asked to reset the conversation."""
    st.header("Session")
    employee = session.get("employee")
    if employee:
        st.success(f"**{employee['name']}** ({employee['employee_id']})  \n"
                   f"{employee['title']} · {employee['department']}  \n📍 {employee['location']}")
    else:
        st.info("No employee verified yet.")
    if session.get("pending_ticket"):
        st.warning("🎫 A ticket draft is awaiting your confirmation.")

    reset = st.button("🔄 New conversation", width="stretch", type="primary")

    st.divider()
    st.subheader("Try asking")
    for prompt in SAMPLE_PROMPTS:
        if st.button(prompt, key=f"sample-{prompt}", width="stretch"):
            st.session_state.queued_prompt = prompt

    with st.expander("Demo employee IDs"):
        for emp_id, desc in DEMO_EMPLOYEES.items():
            st.markdown(f"`{emp_id}` - {desc}")

    with st.expander("Live system status"):
        systems = service.system_status()
        if not systems:
            st.error("System status is unavailable.")
        for system in systems:
            st.markdown(f"{STATUS_ICON.get(system['status'], '⚪')} **{system['system_name']}** - {system['status']}")

    st.divider()
    st.caption(f"Model: `{settings.model}` · tools via MCP ({len(service.tools.llm_tools())})")
    st.caption(f"Conversation ID: `{st.session_state.thread_id[:8]}`")
    return reset


def render_welcome() -> None:
    with st.chat_message("assistant", avatar="🛠️"):
        st.markdown(
            "Hi! I'm your IT support assistant. I can:\n"
            "- answer **how-to** questions from the IT knowledge base,\n"
            "- check the **status of your tickets**,\n"
            "- check whether a **system is down**, and\n"
            "- **raise a new ticket** for you (I'll always confirm before creating it).\n\n"
            "For anything about your tickets I'll need your **employee ID** (e.g. EMP1024)."
        )


def render_message(message: dict[str, Any]) -> None:
    avatar = "🧑‍💼" if message["role"] == "user" else "🛠️"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_provenance(message.get("sources", []))
            _render_trace(message.get("trace", []), message.get("duration_ms"))


def _render_provenance(sources: list[dict[str, str]]) -> None:
    if sources:
        refs = " · ".join(f"{SOURCE_ICON.get(s['kind'], '•')} {s['ref']}" for s in sources)
        st.caption(f"**Retrieved from company systems:** {refs}  \n"
                   "🤖 Wording and any recommendations beyond these records are AI-generated.")
    else:
        st.caption("🤖 AI-generated response - no company records were used.")


def _render_trace(trace: list[dict[str, Any]], duration_ms: int | None) -> None:
    if not trace:
        return
    tools_used = [e["detail"].get("tool") for e in trace if e["kind"] == "tool" and e["detail"].get("tool")]
    title = f"Agent trace · {len(trace)} steps"
    if tools_used:
        title += f" · tools: {', '.join(dict.fromkeys(tools_used))}"
    if duration_ms:
        title += f" · {duration_ms / 1000:.1f}s"
    with st.expander(title):
        for i, event in enumerate(trace, 1):
            icon = KIND_ICON.get(event["kind"], "•")
            st.markdown(f"**{i}. {icon} `{event['node']}`** - {event['label']} "
                        f"<span style='opacity:0.6'>({event['duration_ms']} ms)</span>",
                        unsafe_allow_html=True)
            if event.get("detail"):
                st.json(event["detail"], expanded=False)


def render_pending_ticket(pending: dict[str, Any]) -> str | None:
    """Shows the draft awaiting confirmation. Returns 'yes'/'no' if a button was clicked."""
    draft = pending["draft"]
    with st.container(border=True):
        st.markdown("#### 🎫 Ticket draft - please confirm")
        col1, col2 = st.columns(2)
        col1.markdown(f"**Subject:** {draft['subject']}")
        col1.markdown(f"**Category:** {draft['category']}")
        col2.markdown(f"**Priority:** {draft['priority']}")
        col2.markdown(f"**Employee:** {pending['employee_id']}")
        st.markdown(f"**Description:** {draft['description']}")
        c1, c2, _ = st.columns([2, 1.4, 3])
        if c1.button("✅ Create ticket", type="primary", width="stretch"):
            return "yes"
        if c2.button("❌ Cancel", width="stretch"):
            return "no"
    st.caption("You can also reply in the chat - e.g. 'yes', 'no', or ask for changes.")
    return None
