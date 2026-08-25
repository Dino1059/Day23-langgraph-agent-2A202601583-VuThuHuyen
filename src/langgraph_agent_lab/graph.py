"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any  # noqa: E402

from langgraph.graph import END, START, StateGraph

from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from .routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from .state import AgentState


def build_graph(checkpointer: Any = None) -> Any:  # noqa: ANN401
    """Build and compile the LangGraph StateGraph.

    Architecture:
        START → intake → classify → [conditional: route_after_classify]
          simple       → answer → finalize → END
          tool         → tool → evaluate → [conditional: route_after_evaluate]
                                               success     → answer → finalize → END
                                                needs_retry → retry → [cond: route_after_retry]
                                                                        tool (retry loop)
                                                                        dead_letter → finalize → END
          missing_info → clarify → finalize → END
          risky        → risky_action → approval → [conditional: route_after_approval]
                                                       approved → tool → evaluate → ...
                                                       rejected → clarify → finalize → END
          error        → retry → [conditional: route_after_retry] → tool | dead_letter
    """
    workflow = StateGraph(AgentState)

    # ── 1. Register all 11 nodes ───────────────────────────────────────────────
    workflow.add_node("intake",       intake_node)
    workflow.add_node("classify",     classify_node)
    workflow.add_node("tool",         tool_node)
    workflow.add_node("evaluate",     evaluate_node)
    workflow.add_node("answer",       answer_node)
    workflow.add_node("clarify",      ask_clarification_node)
    workflow.add_node("risky_action", risky_action_node)
    workflow.add_node("approval",     approval_node)
    workflow.add_node("retry",        retry_or_fallback_node)
    workflow.add_node("dead_letter",  dead_letter_node)
    workflow.add_node("finalize",     finalize_node)

    # ── 2. Fixed edges: unconditional transitions ──────────────────────────────
    workflow.add_edge(START,          "intake")
    workflow.add_edge("intake",       "classify")
    workflow.add_edge("tool",         "evaluate")
    workflow.add_edge("risky_action", "approval")
    workflow.add_edge("answer",       "finalize")
    workflow.add_edge("clarify",      "finalize")
    workflow.add_edge("dead_letter",  "finalize")
    workflow.add_edge("finalize",     END)

    # ── 3. Conditional edges: branch based on state values ────────────────────

    # After classify: fan-out to 5 possible routes
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer":      "answer",
            "tool":        "tool",
            "clarify":     "clarify",
            "risky_action":"risky_action",
            "retry":       "retry",
        },
    )

    # After evaluate: success path OR retry loop
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "answer": "answer",
            "retry":  "retry",
        },
    )

    # After retry: bounded retry (go back to tool) OR escalate to dead_letter
    workflow.add_conditional_edges(
        "retry",
        route_after_retry,
        {
            "tool":        "tool",
            "dead_letter": "dead_letter",
        },
    )

    # After approval: proceed with tool OR send back to clarify
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "tool":    "tool",
            "clarify": "clarify",
        },
    )

    # ── 4. Compile with checkpointer (enables persistence / crash-resume) ──────
    return workflow.compile(checkpointer=checkpointer)
