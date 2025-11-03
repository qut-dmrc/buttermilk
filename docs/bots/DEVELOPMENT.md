# Buttermilk Development Instructions

**Load framework DEVELOPMENT.md for generic TDD methodology** (`@$ACADEMICOPS/core/DEVELOPMENT.md`).

This file contains Buttermilk-specific development patterns.

## Mission

Buttermilk aims to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

## Critical Failure Modes (Buttermilk-Specific)

**Six documented patterns that MUST STOP**:

1. **🔴 SECURITY**: NEVER commit real API keys (Vertex AI, Zotero, external APIs)
2. **RUSH-TO-CODE**: Always explore before implementing (read `bots/docs/exploration-before-implementation.md`)
3. **STANDALONE VALIDATION**: No `python -c` testing, only pytest in `tests/`
4. **TUNNEL VISION**: Don't break shared infrastructure (execution_context, llm_core, config)
5. **DEFENSIVE CODING**: Fix root cause, don't add workarounds
6. **REPO POLLUTION**: Use GitHub issues, not tracking markdown files

## Exploration Before Implementation

**MANDATORY**: Read `bots/docs/exploration-before-implementation.md` before implementing features.

**Pattern**:
1. Explore existing code first (Read, Grep, understand architecture)
2. Identify integration points
3. Plan minimal changes
4. Implement one atomic change at a time
5. Test with real flows/data

## Shared Infrastructure

**High-risk Buttermilk files** (extra caution):
- `buttermilk/_core/execution_context.py` - Infrastructure setup
- `buttermilk/_core/llm_core.py` - LLM client abstraction
- `buttermilk/_core/config_bootstrap.py` - Configuration system
- `buttermilk/_core/bm_init.py` - Initialization
- `tests/conftest.py` - Test fixtures

**Before modifying**: Read `bots/docs/impact-analysis.md`.

**Impact Analysis Protocol**:
1. Search all imports: `grep -r "from buttermilk._core.execution_context"`
2. Identify all dependents
3. Consider if change should be in your code instead
4. If truly needed, update ALL dependents
5. Test with real flows to verify nothing breaks

## Buttermilk-Specific Tech Stack

**Core Technologies**:
- **Hydra**: Configuration management (composable configs in `conf/`)
- **Vertex AI**: LLM backend (gemini models)
- **ChromaDB**: Vector storage for RAG
- **FastAPI**: API server for multi-agent flows
- **WebSocket**: Real-time flow orchestration

**Configuration Pattern**:
```python
# ✅ Good - Hydra composable config
# conf/flows/trans.yaml
defaults:
  - /criteria: tja

record_id: ${record_id}
```

**Infrastructure Pattern**:
```python
# ✅ Good - Use execution context
from buttermilk._core.execution_context import from_config_async

execution_context = await from_config_async(config.infrastructure)
```

## Flow Development

**Flows are multi-agent orchestrations** (`buttermilk/flows/`):
- fetch → judge → synth → scorer → diff
- Each agent is atomic, testable
- Flows coordinate via WebSocket messages

**When adding/modifying flows**:
1. Understand existing flow structure (read `flows/` directory)
2. Follow agent naming patterns (FetchAgent, JudgeAgent, etc.)
3. Test with ws_debug_cli (see DEBUGGING.md)
4. Verify flow completes end-to-end with real data

## Data Sources

**Supported Sources**:
- Zotero API (academic literature)
- TMDB (movie data)
- Custom record formats

**Pattern for new data sources**:
1. Implement in `buttermilk/data/sources/`
2. Add corresponding config in `conf/sources/`
3. Create E2E test with real API (see TESTING.md)
4. Document required API keys in `.env.example`

**CRITICAL**: Never commit real API keys. Use environment variables.

See framework DEVELOPMENT.md for generic TDD workflow and quality gates.
