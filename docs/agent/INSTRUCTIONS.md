# Agent Instructions for Buttermilk Project

## Configuration Hierarchy
This is the project-specific configuration for Buttermilk.

1. **Project-Specific** (this file) - Overrides all other configurations
2. **Global** (`/writing/docs/agent/INSTRUCTIONS.md`) - Repository-wide preferences
3. **Base** (`/writing/bot/agents/*.md`) - Core agent definitions

## Buttermilk Mission
Buttermilk makes AI tools accessible for HASS scholars with understandable, traceable, and reproducible workflows.

## 🚨 CRITICAL FAILURE MODES - ZERO TOLERANCE 🚨

**Six documented patterns that MUST STOP:**

1. **🔴 SECURITY BREACH**: NEVER commit real API keys, tokens, passwords, or secrets
2. **RUSH-TO-CODE**: Jumping to implementation without exploration
3. **STANDALONE VALIDATION**: Creating test files outside `tests/` or using `python -c` commands
4. **TUNNEL VISION**: Breaking shared infrastructure to fix specific problems
5. **DEFENSIVE CODING**: Working around failures instead of fixing root causes
6. **REPO POLLUTION**: Creating issue tracking files instead of using GitHub issues

## 🛑 MANDATORY CHECKPOINTS 🛑

### Before ANY file creation:
1. **Security scan**: Contains real credentials? → STOP
2. **Validation check**: Standalone test/demo? → Use proper pytest in `tests/`
3. **Infrastructure check**: Modifying shared files? → MANDATORY impact analysis
   - High-risk: `tests/conftest.py`, `buttermilk/_core/*`, `buttermilk/api/*`, `conf/`
   - NEVER modify shared infrastructure without analyzing ALL dependencies
4. **Repository check**: Progress tracking file? → Use GitHub issues

### Before ANY implementation:
1. **Exploration required**: See [INDEX.md](../bots/INDEX.md) → Rush-to-Code Prevention
2. **Document searches performed** and justify new implementation
3. **Check framework capabilities** before building custom solutions

## 🚨 OBSERVABILITY IS MANDATORY 🚨

**Research integrity principle**: Every run must be fully traceable.
- **NEVER** make tracing/logging optional with defensive code
- **FIX ROOT CAUSE** when observability fails, don't work around it
- **FAIL FAST** if observability cannot be established

**❌ Forbidden patterns:**
```python
# WRONG: Defensive coding around broken infrastructure
if weave_client is not None:
    child_call = weave_client.call(func, *args, **kwargs)
else:
    child_call = None  # HIDES THE FAILURE
```

**✅ Correct approach:**
```python
# RIGHT: Fix root cause or fail fast
weave_client = self.get_weave_client()
if weave_client is None:
    raise RuntimeError("Weave client not available - fix environment setup")
child_call = weave_client.call(func, *args, **kwargs)
```

## Key Technical Context

- **Repository**: @qut-dmrc/buttermilk
- **Architecture**: YAML config → Hydra → OmegaConf → Pydantic → Agent flows
- **Testing**: Use `uv run pytest tests/` - NEVER standalone validation
  - Use `@pytest.mark.anyio` (NOT asyncio)
  - Use provided fixtures: `real_bm`, `real_conf`
  - Test files only in `tests/` directory
- **Configuration**: YAML-only with Hydra/OmegaConf patterns
- **Documentation**: See [../bots/INDEX.md](../bots/INDEX.md) for specialized guidance

## Workflow Enforcement

**For debugging**: ALWAYS read [../bots/debugging.md](../bots/debugging.md) FIRST
**For testing**: NEVER create standalone validation - use existing test patterns
**For shared infrastructure**: MANDATORY impact analysis before changes
**For validation needs**: Use `Task: tester` to create proper pytest tests

**Success = complete flow execution with visible output from ALL agents** (fetch, judge, synth, scorer, diff). Infrastructure fixes are means to this end.