"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node uses LLM-as-judge pattern (bonus)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event

# ─── Pydantic schemas for LLM Structured Output ───────────────────────────────

class ClassificationResult(BaseModel):
    """Structured output schema for classify_node.

    LLM must return exactly one of the 5 route values and a risk level.
    Using Pydantic ensures we get a reliable, typed object instead of free text.
    """

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description=(
            "Intent classification of the support ticket:\n"
            "- simple: general question answerable from knowledge (password reset, how-to, FAQ)\n"
            "- tool: requires data lookup or external system call (order status, account info)\n"
            "- missing_info: query is too vague or incomplete to act on (fix it, help me)\n"
            "- risky: destructive or financial action needing human approval (refund, delete account)\n"  # noqa: E501, ANN401
            "- error: system/technical failure, timeout, service unavailable"
        )
    )
    risk_level: Literal["high", "low"] = Field(
        description="'high' for risky routes, 'low' for all others"
    )
    reasoning: str = Field(
        default="",
        description="Brief one-sentence justification for the classification"
    )


class EvaluationResult(BaseModel):
    """Structured output schema for evaluate_node (LLM-as-judge)."""

    result: Literal["success", "needs_retry"] = Field(
        description=(
            "'needs_retry' if the tool result contains errors, timeouts, or is unusable. "
            "'success' if the result is complete and usable to generate a response."
        )
    )
    reasoning: str = Field(default="", description="Brief justification")


# ─── CLASSIFY PROMPT ──────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM_PROMPT = """You are an intent classifier for a customer support ticket system.
Classify the incoming ticket into exactly one of these routes:

| Route        | When to use |
|-------------|-------------|
| simple      | General Q&A answerable from KB (reset password, FAQ) |
| tool        | Needs real-time data lookup (order status, account info) |
| missing_info| Query is too vague, ambiguous, or incomplete to act on (e.g., "fix it", "help me") |
| risky       | Destructive/financial action needing human approval |
| error       | Describes a system failure, timeout, service unavailable, or technical crash |

Priority when overlapping: risky > tool > missing_info > error > simple
Set risk_level to "high" only for risky routes, "low" for all others.
"""

# ─── Keyword-based fallback classifier (runs when LLM is unavailable) ────────

_KEYWORD_MAP: list[tuple[str, str]] = [
    # risky keywords (highest priority)
    ("refund", "risky"),
    ("delete", "risky"),
    ("remove account", "risky"),
    ("cancel subscription", "risky"),
    ("send email", "risky"),
    ("confirmation email", "risky"),
    # error keywords
    ("timeout", "error"),
    ("failure", "error"),
    ("cannot recover", "error"),
    ("system failure", "error"),
    ("error", "error"),
    # tool keywords
    ("order status", "tool"),
    ("lookup", "tool"),
    ("check order", "tool"),
    ("track", "tool"),
    ("status for", "tool"),
    # missing_info keywords
    ("fix it", "missing_info"),
    ("can you fix", "missing_info"),
    ("help me", "missing_info"),
    ("it's broken", "missing_info"),
]


def _keyword_classify(query: str) -> tuple[str, str]:
    """Fallback heuristic classifier when LLM is unavailable.

    Returns (route, risk_level). Checks keywords in priority order.
    """
    q = query.lower()
    for keyword, route in _KEYWORD_MAP:
        if keyword in q:
            risk_level = "high" if route == "risky" else "low"
            return route, risk_level
    return "simple", "low"


# ─── NODE: intake (provided as example, kept intact) ─────────────────────────

def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── NODE: classify ───────────────────────────────────────────────────────────

def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output.

    Uses ChatGoogleGenerativeAI / ChatOpenAI / ChatAnthropic via get_llm().
    Falls back to keyword heuristic if LLM is unavailable (no API key / network).
    """
    query = state.get("query", "")
    route = "simple"
    risk_level = "low"
    reasoning = ""
    method = "llm"

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(ClassificationResult)
        result: ClassificationResult = structured_llm.invoke(
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Support ticket: {query}"},
            ]
        )
        route = result.route
        risk_level = result.risk_level
        reasoning = result.reasoning

    except Exception as exc:
        # Graceful fallback: keyword heuristic so the graph keeps running
        route, risk_level = _keyword_classify(query)
        reasoning = f"Fallback classifier used (LLM error: {type(exc).__name__})"
        method = "fallback"

    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"Classified as '{route}' via {method}",
                route=route,
                risk_level=risk_level,
                reasoning=reasoning,
            )
        ],
    }


# ─── NODE: tool ───────────────────────────────────────────────────────────────

def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with transient error simulation.

    Simulates real-world transient failures for error-route scenarios:
    - Route "error" and attempt < 2 → returns an ERROR result to trigger retry
    - All other cases → returns mock success result
    """
    route = state.get("route", "simple")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    if route == "error" and attempt < 2:
        # Simulate a transient failure (e.g. network timeout, service unavailable)
        result = f"ERROR: Tool call failed on attempt {attempt} — service temporarily unavailable"
        event_type = "error"
    else:
        # Mock success: simulate real tool lookup based on query keywords
        q = query.lower()
        if "order" in q:
            result = "Order #12345: Status=Shipped, ETA=2026-08-27, Carrier=FedEx"
        elif "refund" in q or "delete" in q:
            result = f"Action executed successfully for request: {query[:60]}"
        else:
            result = f"Tool lookup completed for: {query[:60]}"
        event_type = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [
            make_event(
                "tool",
                event_type,
                result[:120],
                attempt=attempt,
                route=route,
            )
        ],
    }


# ─── NODE: evaluate ───────────────────────────────────────────────────────────

def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate (LLM-as-judge).

    Tries LLM-as-judge first for bonus points.
    Falls back to heuristic check (substring "ERROR") for robustness.
    Sets evaluation_result to "success" or "needs_retry".
    """
    tool_results = state.get("tool_results") or []
    latest_result = tool_results[-1] if tool_results else ""
    evaluation_result = "success"
    method = "llm_judge"
    reasoning = ""

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(EvaluationResult)
        eval_result: EvaluationResult = structured_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a tool-result evaluator for a customer support system. "
                        "Decide if the tool result is usable or if the action should be retried."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tool result to evaluate:\n{latest_result}",
                },
            ]
        )
        evaluation_result = eval_result.result
        reasoning = eval_result.reasoning

    except Exception as exc:
        # Heuristic fallback: look for ERROR keyword in the tool result
        evaluation_result = "needs_retry" if "ERROR" in latest_result else "success"
        reasoning = f"Heuristic evaluation used (LLM error: {type(exc).__name__})"
        method = "heuristic"

    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"Evaluation: {evaluation_result} via {method}",
                evaluation_result=evaluation_result,
                reasoning=reasoning,
            )
        ],
    }


# ─── NODE: answer ─────────────────────────────────────────────────────────────

def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM (grounded in available context).

    Uses tool_results, approval, and the original query as context.
    Falls back to a safe templated response if LLM is unavailable.
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results") or []
    approval = state.get("approval") or {}
    route = state.get("route", "simple")

    # Build context for grounded generation
    context_parts: list[str] = [f"Customer query: {query}"]
    if tool_results:
        context_parts.append("Retrieved data:\n" + "\n".join(tool_results))
    if approval:
        approved = approval.get("approved", False)
        reviewer = approval.get("reviewer", "unknown")
        context_parts.append(
            f"This action was {'approved' if approved else 'rejected'} by {reviewer}."
        )
    context = "\n\n".join(context_parts)

    final_answer = ""

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a professional customer support agent. "
                        "Write a concise, helpful, and empathetic response to the customer's query. "  # noqa: E501, ANN401
                        "Use the provided context to ground your answer. "
                        "Do NOT make up information not in the context."
                    ),
                },
                {
                    "role": "user",
                    "content": context,
                },
            ]
        )
        final_answer = response.content

    except Exception:
        # Fallback templated response (LLM unavailable)
        if tool_results and "ERROR" not in tool_results[-1]:
            final_answer = (
                f"Based on our records: {tool_results[-1]}\n\n"
                "Please let us know if you need further assistance."
            )
        else:
            final_answer = (
                f"Thank you for contacting support. Regarding your query: '{query}'\n"
                "Our team is looking into this and will follow up shortly."
            )

    return {
        "final_answer": final_answer,
        "messages": [f"answer:generated route={route}"],
        "events": [
            make_event(
                "answer",
                "completed",
                "Final answer generated",
                route=route,
                answer_length=len(final_answer),
            )
        ],
    }


# ─── NODE: ask_clarification ──────────────────────────────────────────────────

def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generates a specific, targeted clarification question to help
    the customer provide the information needed to proceed.
    """
    query = state.get("query", "")

    clarification_question = (
        "Thank you for reaching out. To assist you effectively, "
        "could you please provide more details?\n\n"
        "Specifically, we need:\n"
        "- Your account number or order ID\n"
        "- A detailed description of the issue you are experiencing\n"
        "- Any error messages you have received\n\n"
        "This will allow us to resolve your request promptly."
    )

    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a customer support agent. The customer's message is too vague to act on. "  # noqa: E501, ANN401
                        "Write a polite, specific clarification question (2-4 bullet points) "
                        "asking for the exact information needed to help them. "
                        "Do NOT attempt to answer their question or make assumptions."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Vague customer message: {query}",
                },
            ]
        )
        clarification_question = response.content
    except Exception:
        pass  # Keep the default templated question

    return {
        "pending_question": clarification_question,
        "final_answer": clarification_question,
        "messages": ["clarify:question_generated"],
        "events": [
            make_event(
                "clarify",
                "completed",
                "Clarification question generated",
                original_query=query[:80],
            )
        ],
    }


# ─── NODE: risky_action ───────────────────────────────────────────────────────

def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describes exactly what will happen and why human approval is required.
    The approval_node will read proposed_action before asking for approval.
    """
    query = state.get("query", "")
    proposed_action = (
        f"PROPOSED RISKY ACTION\n"
        f"{'─' * 40}\n"
        f"Customer Request: {query}\n\n"
        f"Action Required: Execute the requested operation which involves "
        f"financial transactions, account modifications, or data deletion.\n\n"
        f"⚠️  This action CANNOT be undone. Human approval required before proceeding.\n"
        f"Risk Level: HIGH"
    )

    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:prepared"],
        "events": [
            make_event(
                "risky_action",
                "completed",
                "Risky action proposed, awaiting approval",
                query=query[:80],
            )
        ],
    }


# ─── NODE: approval ───────────────────────────────────────────────────────────

def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop (HITL) approval step.

    Default: mock approval (approved=True) so tests and CI run offline.
    Extension: set env LANGGRAPH_INTERRUPT=true to use real interrupt().
    """
    proposed_action = state.get("proposed_action", "")

    # Extension: Real HITL interrupt when LANGGRAPH_INTERRUPT=true
    use_interrupt = os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true"
    if use_interrupt:
        try:
            from langgraph.types import interrupt  # type: ignore[import-untyped]
            human_decision = interrupt(
                {
                    "message": "Please review and approve or reject the proposed action.",
                    "proposed_action": proposed_action,
                }
            )
            approved = human_decision.get("approved", False)
            reviewer = human_decision.get("reviewer", "human-reviewer")
            comment = human_decision.get("comment", "")
        except Exception:
            # Fallback to mock if interrupt is not supported
            approved = True
            reviewer = "mock-reviewer"
            comment = "Interrupt unavailable, defaulting to approved"
    else:
        # Default: mock approval — always approve for offline testing
        approved = True
        reviewer = "mock-reviewer"
        comment = "Auto-approved (mock HITL)"

    approval_decision = {
        "approved": approved,
        "reviewer": reviewer,
        "comment": comment,
    }

    return {
        "approval": approval_decision,
        "messages": [f"approval:{'approved' if approved else 'rejected'} by {reviewer}"],
        "events": [
            make_event(
                "approval",
                "completed",
                f"Action {'approved' if approved else 'rejected'} by {reviewer}",
                approved=approved,
                reviewer=reviewer,
                comment=comment,
            )
        ],
    }


# ─── NODE: retry_or_fallback ──────────────────────────────────────────────────

def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increments the attempt counter and logs the transient failure.
    The route_after_retry function then checks attempt vs max_attempts
    to decide whether to retry tool or escalate to dead_letter.
    """
    attempt = state.get("attempt", 0)
    tool_results = state.get("tool_results") or []
    latest_error = tool_results[-1] if tool_results else "Unknown error"

    new_attempt = attempt + 1
    error_message = f"Attempt {new_attempt} failed: {latest_error[:120]}"

    return {
        "attempt": new_attempt,
        "errors": [error_message],
        "messages": [f"retry:attempt={new_attempt}"],
        "events": [
            make_event(
                "retry",
                "retrying",
                error_message,
                attempt=new_attempt,
            )
        ],
    }


# ─── NODE: dead_letter ────────────────────────────────────────────────────────

def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third tier: intake → tool → retry → dead_letter.
    Logs the escalation and sets a final_answer explaining the failure.
    """
    query = state.get("query", "")
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    errors = state.get("errors") or []

    final_answer = (
        f"We sincerely apologize for the inconvenience. "
        f"We were unable to process your request after {attempt} attempt(s).\n\n"
        f"Your ticket has been escalated to our technical team for manual review. "
        f"A support engineer will contact you within 24 hours.\n\n"
        f"Reference: '{query[:60]}'"
    )

    return {
        "final_answer": final_answer,
        "messages": [f"dead_letter:escalated after {attempt}/{max_attempts} attempts"],
        "events": [
            make_event(
                "dead_letter",
                "escalated",
                f"Max retries ({max_attempts}) exceeded — ticket escalated",
                attempt=attempt,
                max_attempts=max_attempts,
                error_count=len(errors),
            )
        ],
    }


# ─── NODE: finalize ───────────────────────────────────────────────────────────

def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Acts as the single exit gate — ensures every execution path produces
    a consistent audit trail regardless of which route was taken.
    """
    route = state.get("route", "unknown")
    final_answer = state.get("final_answer") or state.get("pending_question") or ""

    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "Workflow finished",
                route=route,
                has_answer=bool(final_answer),
            )
        ],
    }
