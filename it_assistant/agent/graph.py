"""LangGraph workflow definition.

    START
      │  route_entry: is a ticket draft awaiting confirmation?
      ├──────────────► confirmation_gate ──(yes)──► commit_ticket ─┐
      │                       │ (no / unclear)                    │
      ▼                       ▼                                   ▼
    ingest_user_message ◄─────────────────────────────────────────┘
      ▼
    agent (LLM: pick a tool or answer) ◄─────────────────────────┐
      │  route_after_agent                                       │
      ├── employee_lookup ───────────────────────────────────────┤
      ├── knowledge_search ──────────────────────────────────────┤
      ├── ticket_lookup ─────────────────────────────────────────┤
      ├── ticket_creation (validate, dedupe, stage) ─────────────┤
      ├── system_status ─────────────────────────────────────────┤
      ├── unknown_tool ──────────────────────────────────────────┘
      ▼
    respond (verify ticket IDs) ──► END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from it_assistant.agent import nodes as n
from it_assistant.agent.state import AgentState


def build_graph(agent_nodes: n.AgentNodes, checkpointer=None):
    graph = StateGraph(AgentState)

    graph.add_node(n.CONFIRMATION_GATE, agent_nodes.confirmation_gate)
    graph.add_node(n.COMMIT_TICKET, agent_nodes.commit_ticket)
    graph.add_node(n.INGEST, agent_nodes.ingest_user_message)
    graph.add_node(n.AGENT, agent_nodes.agent)
    graph.add_node(n.EMPLOYEE_LOOKUP, agent_nodes.employee_lookup)
    graph.add_node(n.KNOWLEDGE_SEARCH, agent_nodes.knowledge_search)
    graph.add_node(n.TICKET_LOOKUP, agent_nodes.ticket_lookup)
    graph.add_node(n.TICKET_CREATION, agent_nodes.ticket_creation)
    graph.add_node(n.SYSTEM_STATUS, agent_nodes.system_status)
    graph.add_node(n.UNKNOWN_TOOL, agent_nodes.unknown_tool)
    graph.add_node(n.RESPOND, agent_nodes.respond)

    graph.add_conditional_edges(START, n.route_entry, [n.CONFIRMATION_GATE, n.INGEST])
    graph.add_conditional_edges(n.CONFIRMATION_GATE, n.route_after_confirmation, [n.COMMIT_TICKET, n.INGEST])
    graph.add_edge(n.COMMIT_TICKET, n.INGEST)
    graph.add_edge(n.INGEST, n.AGENT)

    tool_nodes = [
        n.EMPLOYEE_LOOKUP, n.KNOWLEDGE_SEARCH, n.TICKET_LOOKUP,
        n.TICKET_CREATION, n.SYSTEM_STATUS, n.UNKNOWN_TOOL,
    ]
    graph.add_conditional_edges(n.AGENT, n.route_after_agent, [*tool_nodes, n.RESPOND])
    for node in tool_nodes:
        graph.add_edge(node, n.AGENT)  # tool result goes back to the LLM (multi-step loop)

    graph.add_edge(n.RESPOND, END)
    return graph.compile(checkpointer=checkpointer)
