# Day 08 Lab Report

## 1. Team / Student

- **Name:** vuthuhuyen
- **Date:** 2026-08-25 17:05
- **LLM Provider:** Google Gemini 2.5 Flash (via `langchain-google-genai`)
- **Checkpointer:** MemorySaver (memory) + SQLite evidence (outputs/checkpoints.db)

---

## 2. Architecture

The graph implements a **Support-Ticket Agent** with 11 nodes and 4 conditional routing functions.

### Node Responsibilities

| Node | Role |
|---|---|
| `intake` | Normalize and sanitize the raw customer query |
| `classify` | LLM + Pydantic structured output → classify into 5 route intents |
| `tool` | Mock tool execution with transient error simulation for retry testing |
| `evaluate` | LLM-as-judge → assess tool result quality (`success` / `needs_retry`) |
| `answer` | LLM grounded response generation using tool results and approval context |
| `clarify` | Generate targeted clarification question for vague/incomplete queries |
| `risky_action` | Prepare description of destructive/financial action for approval |
| `approval` | HITL gate — mock approval by default, supports `LANGGRAPH_INTERRUPT=true` |
| `retry` | Increment attempt counter, log error, gate for bounded retry loop |
| `dead_letter` | Escalate unresolvable failures after max retries exceeded |
| `finalize` | Single exit gate — audit event for all routes before `END` |

### Graph Routing

```text
START → intake → classify → [route_after_classify]
  simple        → answer → finalize → END
  tool          → tool → evaluate → [route_after_evaluate]
                              success     → answer → finalize → END
                              needs_retry → retry  → [route_after_retry]
                                              attempt < max  → tool (loop)
                                              attempt >= max → dead_letter → finalize → END
  missing_info  → clarify → finalize → END
  risky         → risky_action → approval → [route_after_approval]
                                    approved → tool → evaluate → ...
                                    rejected → clarify → finalize → END
  error         → retry → [route_after_retry] → tool | dead_letter
```

---

## 3. State Schema

| Field | Reducer | Why |
|---|---|---|
| `thread_id` | overwrite | Unique key for checkpointing per session |
| `scenario_id` | overwrite | Identifier for grading |
| `query` | overwrite | Current normalized customer query |
| `route` | overwrite | Active route — only current value matters |
| `risk_level` | overwrite | `"high"` or `"low"` — changes per classification |
| `attempt` | overwrite | Current retry count — incremented in-place |
| `max_attempts` | overwrite | Retry ceiling — set once from scenario config |
| `final_answer` | overwrite | Last generated answer replaces previous |
| `pending_question` | overwrite | Clarification question — one at a time |
| `proposed_action` | overwrite | Risky action description for approval review |
| `approval` | overwrite | HITL decision dict — one per risky flow |
| `evaluation_result` | overwrite | `"success"` or `"needs_retry"` — retry gate |
| `messages` | **append** (`add`) | Full message audit log — must never be lost |
| `tool_results` | **append** (`add`) | All tool call results — needed for retry diagnosis |
| `errors` | **append** (`add`) | Error log per retry — grading and debugging |
| `events` | **append** (`add`) | Structured audit events — grading evidence |

---

## 4. Scenario Results

**Summary:** 7/7 scenarios passed · Success rate: 100.0%
Total retries: 3 · Total HITL interrupts: 2
Average nodes visited: 6.4

| Scenario | Expected Route | Actual Route | Success | Retries | Interrupts |
|---|---|---|:---:|---:|---:|
| S01_simple | simple | simple | ✅ | 0 | 0 |
| S02_tool | tool | tool | ✅ | 0 | 0 |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 |
| S04_risky | risky | risky | ✅ | 0 | 1 |
| S05_error | error | error | ✅ | 2 | 0 |
| S06_delete | risky | risky | ✅ | 0 | 1 |
| S07_dead_letter | error | error | ✅ | 1 | 0 |

---

## 5. Failure Analysis

### Failure Mode 1 — Transient Tool Failure & Bounded Retry Loop

**Scenario:** `S05_error` and `S07_dead_letter`

**What can go wrong:** External APIs (order systems, payment gateways) suffer transient
timeouts or 5xx errors. Without a retry strategy, a single failure causes a permanent
bad user experience. Without a **bound** on retries, infinite loops drain quota and money.

**How we handle it:**
- `tool_node` returns a result containing `"ERROR"` when `route == "error"` and `attempt < 2`.
- `evaluate_node` (LLM-as-judge) detects the error → sets `evaluation_result = "needs_retry"`.
- `route_after_evaluate` sends execution to `retry` node.
- `retry_or_fallback_node` increments `attempt` by 1.
- `route_after_retry` checks `attempt < max_attempts` → back to `tool`; otherwise → `dead_letter`.
- `S07_dead_letter` sets `max_attempts=1` so the loop exhausts immediately.

**Key design principle:** The retry counter lives in state (overwrite field) so the
checkpointer persists it across possible process restarts — true crash-resume.

---

### Failure Mode 2 — Risky Action Without Human Approval

**Scenario:** `S04_risky` and `S06_delete`

**What can go wrong:** An AI agent executing financial refunds or account deletions
autonomously — without oversight — is a major production safety risk. False positives
(hallucinated confidence) can cause irreversible data loss or financial damage.

**How we handle it:**
- `classify_node` detects destructive/financial intent → `route = "risky"`, `risk_level = "high"`.
- `risky_action_node` prepares a clear description of the proposed action.
- `approval_node` is a **mandatory HITL gate**: in production (`LANGGRAPH_INTERRUPT=true`),
  it calls `langgraph.types.interrupt()` to pause the graph and wait for a human decision.
- Only after `approved=True` does the graph proceed to `tool` → `evaluate` → `answer`.
- If rejected, the graph routes to `clarify` — asking the customer for an alternative.

---

## 6. Persistence & Recovery Evidence

**Checkpointer:** `MemorySaver` (default) + `SqliteCheckpointer` (stdlib `sqlite3`, WAL mode).

**How it works:**
- Every scenario run is assigned a unique `thread_id` (e.g., `thread-S01_simple`).
- `MemorySaver` stores state snapshots in-process after each node.
- `SqliteCheckpointer` writes JSON state snapshots to `outputs/checkpoints.db` using WAL mode.
- WAL (Write-Ahead Logging) allows concurrent reads during writes — production-safe.

**Crash-resume capability:**
```python
checkpointer = build_checkpointer("sqlite")
graph = build_graph(checkpointer=checkpointer)
# If process crashes mid-execution, resume with same thread_id:
result = graph.invoke(state, config={"configurable": {"thread_id": "thread-S04_risky"}})
```
The graph will resume from the last persisted checkpoint instead of restarting from `START`.

**State history inspection:**
```python
sqlite_ckpt = SqliteCheckpointer("outputs/checkpoints.db")
history = sqlite_ckpt.get_history("thread-S04_risky")
# Returns list of {step, state, ts} — full audit trail
```

---

## 7. Extension Work

| Extension | Status | Details |
|---|---|---|
| SQLite Persistence | ✅ | Custom `SqliteCheckpointer` with WAL mode + `HybridSaver` |
| LLM-as-Judge | ✅ | `evaluate_node` uses `EvaluationResult` structured output |
| Mermaid Graph Diagram | ✅ | Auto-generated via `graph.get_graph().draw_mermaid()` |
| HITL Interrupt Support | ✅ | `approval_node` checks `LANGGRAPH_INTERRUPT=true` env var |
| Graceful LLM Fallback | ✅ | All LLM nodes fall back to heuristics when offline |

---

## 8. Improvement Plan

If I had one more day to productionize this system, I would prioritize:

1. **Streaming responses:** Replace `graph.invoke()` with `graph.stream()` so partial
   answers appear progressively in the UI — critical for user experience with slow LLMs.

2. **Real HITL with Streamlit UI:** Build a minimal approval UI with `st.button("Approve")`
   / `st.button("Reject")` that calls `graph.invoke(Command(resume=...))` with the
   reviewer's decision, storing it in a Postgres-backed checkpointer.

3. **Postgres persistence + pgvector:** Replace SQLite with Postgres for production-grade
   multi-process concurrency. Add pgvector for semantic search over historical tickets
   to provide few-shot examples to `classify_node` and `answer_node`.

4. **Observability:** Integrate LangSmith tracing for every LLM call — latency, token
   cost, and classification accuracy per route — enabling continuous prompt improvement.
