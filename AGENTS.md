# Repository Guidelines

## Project Structure & Module Organization

- `src/langgraph_agent_lab/` contains the installable Python package: state models, graph wiring, nodes, routing, persistence, metrics, reporting, and the Typer CLI.
- `tests/` contains pytest unit, routing, state, metrics, and graph smoke tests.
- `configs/` stores YAML run configurations; `data/sample/` contains JSONL scenarios.
- `docs/` contains the lab guide, rubric, and metrics documentation. Reports belong in `reports/`; generated metrics belong in `outputs/`.

Keep workflow behavior in the package rather than in the CLI. Preserve serializable LangGraph state and append-only audit events when adding nodes.

## Build, Test, and Development Commands

```bash
make install       # Install the project and development dependencies
make test          # Run the pytest suite
make lint          # Run Ruff on src/ and tests/
make typecheck     # Run mypy on src/
make run-scenarios # Run configured scenarios and write outputs/metrics.json
make grade-local   # Validate the generated metrics file
make clean         # Remove caches and generated build/metric artifacts
```

Use a Python 3.11 virtual environment. Scenario runs and graph smoke tests require an installed provider extra (for example, `pip install -e '.[openai,dev]'`) and an API key in `.env`.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations, and focused functions. Ruff enforces a 100-character line limit and the configured `E`, `F`, `I`, `B`, `UP`, `N`, and `ANN` rules. Use `snake_case` for modules, functions, and variables; `PascalCase` for Pydantic models and other types; and descriptive `*_node` names for graph nodes. Do not mutate the incoming state inside node functions—return partial updates.

## Testing Guidelines

Tests use pytest and live under `tests/`, with `test_*.py` files and `test_*` functions. Run `make test` before submitting changes; run `make lint` and `make typecheck` as well. Graph smoke tests are skipped when no LLM API key is configured. Add or update tests for routing, state changes, retry bounds, and any new node behavior; no fixed coverage threshold is configured.

## Commit & Pull Request Guidelines

Match the existing history with concise, imperative commit subjects (for example, `add retry routing tests`). Pull requests should explain the behavior change, link the relevant issue or lab task, and report `make test`, `make lint`, and `make typecheck` results. Include sample `make run-scenarios` or report output when workflow or metrics behavior changes. Never commit `.env`, API keys, checkpoints, or hidden grading data.
