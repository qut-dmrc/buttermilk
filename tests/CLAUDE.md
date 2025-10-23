# Testing Instructions for buttermilk

Load testing methodology and standards when working in tests/:

## Core Testing Documentation

- @../bots/docs/_CHUNKS/E2E-TESTING.md - TRUE end-to-end testing (real APIs, real storage)
- @../.claude/skills/test-writing/SKILL.md - Complete test-writing methodology
- @../bots/agents/_CORE.md - Core axioms (fail-fast, DRY, etc.)

## Key Principles

1. **TRUE E2E tests** use REAL APIs, REAL storage, REAL processors
   - NO mocks of our internal code
   - Mock only at system boundaries (external APIs)
   - Use `real_bm` fixture for configuration

2. **Test-Driven Development**
   - Write failing test first
   - Implement minimum code
   - Test passes
   - Refactor

3. **Integration over Unit**
   - Test complete workflows
   - Use real configurations
   - Delete brittle unit tests
   - Consolidate 10 unit tests → 1 integration test

4. **Real Data, Not Fakes**
   - Load from JSON fixtures
   - No inline fake data
   - Use realistic test data

## Test Categories

**Unit Tests** (`tests/`):
- Isolated components
- Demonstrate bugs
- Verify specific fixes

**Integration Tests** (`tests/`):
- 2-3 components working together
- Real configuration
- May mock external APIs

**TRUE E2E Tests** (`tests/data/test_*_e2e.py`):
- Complete pipeline start to finish
- REAL everything (APIs, storage, processors)
- Validates production workflow
- See E2E-TESTING.md for patterns

## Required Fixtures

From `conftest.py`:

- `real_bm` - Fully configured BM instance (session-scoped)
- `real_conf` - Raw Hydra DictConfig (session-scoped)
- `real_llms` - LLMs instance
- `real_llm` - Cheap chat model instance
- `tmp_path` - Temporary directory (pytest built-in)

## Test File Naming

- `test_*.py` - Standard tests
- `test_*_e2e.py` - TRUE end-to-end tests (real APIs)
- `test_*_integration.py` - Integration tests (2+ components)

## Running Tests

```bash
# Run specific test
uv run pytest tests/test_file.py::test_name -xvs

# Run all tests in file
uv run pytest tests/test_file.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run E2E tests only
uv run pytest tests/data/test_*_e2e.py -v

# Run excluding slow tests
uv run pytest tests/ -m "not slow"
```

## Anti-Patterns to Avoid

❌ **FORBIDDEN**:
- Mocking internal code (`@patch("buttermilk.*")`)
- Loading Hydra configs in test files
- Inline fake data
- Testing implementation details
- Creating config-loading fixtures

✅ **REQUIRED**:
- Use `real_bm` or `real_conf` fixtures
- Load test data from JSON files
- Test business behavior
- REAL API calls in E2E tests
- REAL storage (temp locations OK)
