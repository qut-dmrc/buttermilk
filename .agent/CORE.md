# Buttermilk Agent Instructions

Buttermilk aims to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

- Use "uv" to manage dependencies. Install with  `uv sync --extra dev --upgrade`
- Tests are classified by marker; "endtoend" and "slow" and "demo" and "integration".
- **Debugging and Log Analysis**: Use `ws_debug_cli` for all log analysis. See `.agent/workflows/DEBUGGING.md` for the authoritative Golden Path.
  - **Analyze logs**: `uv run python -m buttermilk.debug.ws_debug_cli analyze --file <path>`
  - **View logs**: `uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 --file <path>`
  - **Live debugging**: Use `ws_debug_cli start ...` (see Golden Path guide).
  - Note: `/tmp/bm_*` files are **LOG** files, not execution traces. Do not use `trace_analysis` on them.
- our framework goal is success first time, every time, with just-in-time information, that doesn't cause unecessary cost or delay.

### Testing Philosophy

**Mock only at system boundaries**:
- Network calls, filesystem, time, environment variables
- NEVER mock `buttermilk.*` code or business logic

See `.agent/workflows/TESTING.md` for complete guide.

## Observability Requirements

**Research integrity depends on complete observability**. Logging, tracing, and data saving MUST work correctly.

- Fail fast, fail loud. Always.
- NO WORKAROUNDS. Always raise errors and HALT -- do not be selfish and hide errors you discover from other researchers.

**Never**:
- Add defensive code around tracing failures (`if x is not None:`)
- Use default values for configuration (`config_x = cfg.get('x', 'default')`)
- Make observability optional when it should be required
- Silently continue when logging/saving fails
- Work around broken infrastructure instead of fixing it

## Technical Stack

**Configuration**: Hydra (composable YAML configs in `conf/`)
**LLM Backend**: Vertex AI (Gemini models)
**Vector Storage**: ChromaDB for RAG
**API**: FastAPI with WebSocket for multi-agent flows
**Testing**: pytest with real infrastructure via `real_bm` fixture
