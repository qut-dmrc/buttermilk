# Test Fixer Agent Instructions

You are responsible for maintaining quality assurance and testing in the project.

## Testing Philosophy

We have several layers of tests: unit, integration, end-to-end.

**No Standalone Test Scripts**:

- All tests must be in the `tests/` directory and use `pytest`.
- Tests should be written to be reused. No single-use testing.

**ALWAYS** use existing fixtures:

- The main `tests/conftest.py` initialises the main BM singleton and other resources.
- ALL test configuration is located in `conf/testing.yaml`. NEVER write your own configuration dictionaries.
- ALWAYS rely on the pytest fixtures to load configurations with hydra. NEVER load your own configurations.
- ALWAYS use the 'real_bm', 'real_logger', 'real_llm' etc fixtures.
- Each test layer has its own 'conftest.py' that provides any specifically required fixtures.

**All tests MUST work**:

- Don't skip tests or create simpler alternatives.
- If a test is overly complicated or out of date, it is YOUR responsibility to fix it.
- Never change validation rules (e.g. Pydantic's `extra="forbid"`) to hide errors.
- No shortcuts.

**Mock only at system boundaries** (network, filesystem, time, env):

- Mock ONLY: network, filesystem, time, env, randomness
- NEVER mock: buttermilk.* (our code)
- See [TESTING_PHILOSOPHY.md](TESTING_PHILOSOPHY.md) for details.

**Other notes**:

- Use `@pytest.mark.anyio` for async tests.

## COMMANDS REFERENCE

1. **ALWAYS** run ruff first - it finds problems faster than reading code:

```bash
# See all problems and auto-fix trivial issues
uv run ruff check --output-format=concise --fix tests/[category]/
```

2. Use **uv run pytest** to test: `uv run pytest ...`
