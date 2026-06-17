# Available Documentation

Orientation index for agents working in the buttermilk repo. Paths are relative to the repo root and verified to exist.

## Agent doctrine

- **`.agent/CORE.md`**: Buttermilk agent operating instructions — path discovery, fail-fast/halt rule, observability requirements, methodology-belongs-to-researcher, technical stack
- **`.agent/rules/`**: Repo coding rules — `config.md` (Hydra config), `STYLE.md` (code style), `techstack.md` (stack conventions), `tests.md` (test rules)
- **`.agent/workflows/`**: Authoritative golden-path workflows — `DEBUGGING.md` (ws_debug_cli log analysis), `TESTING.md`, `DEVELOPMENT.md`, `GROUPCHAT-DEBUG.md`, `PERFORMANCE_TESTING.md`
- **`CLAUDE.md` / `AGENTS.md` / `GEMINI.md`**: Thin entry points that import `.agent/CORE.md` and point at rules/workflows

## Project overview

- **`README.md`**: Project overview — core concepts (Flows, Records, Pipelines, Processors, Orchestrators, Agents), features, installation, quickstart, public Python API surface
- **`CONTRIBUTING.md`**: Development setup (uv), test markers, design-first workflow, release process
- **`docs/README.md`**: Documentation landing page — points to README, examples, and conf as the most reliable current references
- **`CHANGELOG.md`**: Release history
- **`examples/`**: Runnable examples and notebooks (flows, batch, visualisation, image analysis); see `examples/README.md` and `examples/VISUALIZATION.md`

## Source layout (`buttermilk/`)

- **`buttermilk/_core/`**: Core framework internals — agent base classes, config bootstrap, contracts, orchestrator, messages, exceptions
- **`buttermilk/_core/llms.py`**: LLM client/model registry and routing — the canonical model-selection and provider-wiring module
- **`buttermilk/agents/`**: Agent implementations (LLM agents, scrapers, classifiers, evaluators)
- **`buttermilk/orchestrators/`**: Flow orchestration — coordinate multi-agent groupchat execution
- **`buttermilk/runner/`**: Flow runners and execution entry points
- **`buttermilk/api/`**: FastAPI + WebSocket API for multi-agent flows
- **`buttermilk/storage/`**: Storage backends (BigQuery, ChromaDB vector storage, caching)
- **`buttermilk/processors/`**: Composable processors (async iterators over `Record`s)
- **`buttermilk/templates/`**: Prompt templating system
- **`buttermilk/debug/`**: Debugging tooling — `ws_debug_cli` (run as `python -m buttermilk.debug.ws_debug_cli`), log analysis, trace analysis
- **`buttermilk/utils/`**: Utility modules (formatting, BigQuery, OTel, templating, validators)
- **`buttermilk/utils/pricing.py`**: Token-cost utilities — LLM usage cost calculation (lazy-loads litellm cost calculator)
- **`buttermilk/cli/`**: Command-line entry points
- **`buttermilk/batch/`**: Batch processing for large-scale data work
- **`buttermilk/monitoring/`**: Observability and monitoring
- **`buttermilk/toxicity/`**: Toxicity/content-classification components

## Configuration (`buttermilk/conf/`)

- **`buttermilk/conf/config.yaml`**: Root Hydra config — composable defaults
- **`buttermilk/conf/llms/`**: LLM profile configs — `full.yaml` (all models), `lite.yaml`, `expensive.yaml`, `debug.yaml`, `test_safe.yaml`
- **`buttermilk/conf/flows/`**: Flow definitions
- **`buttermilk/conf/agents/`**: Agent configs
- **`buttermilk/conf/storage/`**: Storage backend configs
- **`buttermilk/conf/run/`** and **`buttermilk/conf/tools/`**: Run modes and tool configs

## Tests (`tests/`)

- **`tests/CLAUDE.md`**: Testing instructions and methodology entry point for work under `tests/`
- **`tests/conftest.py`**: Shared fixtures (including the `real_bm` real-infrastructure fixture)
- **`tests/`**: Test suite organised by area — `agents/`, `api/`, `orchestrators/`, `storage/`, `endtoend/`, `integration/`, `performance/`, `unit/`, plus others. Tests are classified by marker: `endtoend`, `slow`, `demo`, `integration`
