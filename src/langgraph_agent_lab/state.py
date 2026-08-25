"""State schema for the Day 08 LangGraph lab.

Students should extend the schema only when needed. Keep state lean and serializable.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph state for the support-ticket agent.

    Field design decisions:
    - Overwrite fields: single current value that changes as the workflow progresses.
    - Append-only fields (Annotated[list, add]): audit trails that must never be lost.
    """

    # ── Core ticket fields (overwrite) ────────────────────────────────────
    thread_id: str            # Unique ID for checkpointing / persistence
    scenario_id: str          # Scenario identifier for grading
    query: str                # Normalized user query
    route: str                # Classified route: simple/tool/missing_info/risky/error
    risk_level: str           # "high" for risky routes, "low" otherwise

    # ── Retry tracking (overwrite) ─────────────────────────────────────────
    attempt: int              # Current retry attempt count (starts at 0)
    max_attempts: int         # Maximum retries allowed (default 3)

    # ── Output fields (overwrite) ──────────────────────────────────────────
    final_answer: str | None  # Final response sent to the user
    pending_question: str     # Clarification question when missing_info route

    # ── Risky action flow (overwrite) ─────────────────────────────────────
    proposed_action: str      # Description of risky action proposed for approval

    # ── Human-in-the-loop approval (overwrite) ────────────────────────────
    # Stored as a plain dict for JSON serializability (not ApprovalDecision model)
    approval: dict[str, Any] | None  # {"approved": bool, "reviewer": str, "comment": str}

    # ── Retry-loop gate (overwrite) ───────────────────────────────────────
    # Drives the route_after_evaluate conditional edge
    evaluation_result: str    # "success" or "needs_retry"

    # ── Append-only audit / history fields ────────────────────────────────
    messages: Annotated[list[str], add]             # Human-readable log messages
    tool_results: Annotated[list[str], add]         # Raw tool call results (one per call)
    errors: Annotated[list[str], add]               # Error strings (one per failure)
    events: Annotated[list[dict[str, Any]], add]    # Structured audit events (LabEvent)


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario.

    Every field declared in AgentState is initialized here so that
    node functions can safely call state.get("field") without KeyError.
    """
    return {
        # Core ticket fields
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        # Retry tracking
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        # Output fields
        "final_answer": None,
        "pending_question": "",
        # Risky action
        "proposed_action": "",
        # HITL approval
        "approval": None,
        # Retry-loop gate
        "evaluation_result": "",
        # Append-only audit lists
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: Any) -> dict[str, Any]:  # noqa: E501, ANN401
    """Create a normalized event payload."""
    return LabEvent(
        node=node, event_type=event_type, message=message, metadata=metadata
    ).model_dump()
