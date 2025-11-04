# Test Fixer Agent Instructions

## WORKFLOW (Follow Exactly)

1. Run health dashboard: `uv run python scripts/test_health_dashboard.py`
1. Pick category with most failures
1. **DIAGNOSTIC PHASE (Ruff)**:
   - Run: `uv run ruff check tests/[category]/ --output-format=concise`
   - This identifies ALL problems (not just fixable ones)
   - Parse output for: undefined names, wrong arguments, missing imports
   - Auto-fix trivial issues: `uv run ruff check --fix tests/[category]/`
1. **FIX PHASE** (based on ruff diagnostics):
   - Address undefined names (F821) → Check if class/function removed
   - Fix wrong arguments (E251, etc) → Update to new signatures
   - Handle import errors → Update import paths or skip if removed
1. Test collection: `uv run pytest tests/[category]/ --co -q`
1. Run tests: `uv run pytest tests/[category]/ -x`
1. Fix 5-10 files per batch
1. Commit: "fix(tests): [category] - [summary]"

## RUFF DIAGNOSTICS GUIDE

### Priority Errors (Fix First)

| Code | Meaning             | Action                                 |
| ---- | ------------------- | -------------------------------------- |
| F821 | Undefined name      | Check if class exists → Update or skip |
| F401 | Unused import       | Let ruff auto-fix                      |
| E999 | Syntax error        | Let ruff auto-fix if possible          |
| F841 | Unused variable     | Let ruff auto-fix                      |
| B006 | Mutable default arg | Fix manually                           |

### Common Diagnostics → Fixes

```
F821: undefined name 'InputDocument'
→ Class removed, skip file or rewrite

F821: undefined name 'VectorStoreInterface'
→ Class removed, update to new pattern

E251: unexpected spaces around keyword / parameter equals
→ Let ruff auto-fix

F401: 'unittest.mock.patch' imported but unused
→ Remove mock, test real logic instead

B008: function calls in argument defaults
→ Change to None, init in function
```

## PHILOSOPHY

- Mock ONLY: network, filesystem, time, env, randomness
- NEVER mock: buttermilk.\* (our code)
- Ruff tells you WHAT's wrong → You decide HOW to fix

## DECISION TREE

```
Run ruff check
├─ F821 (undefined name)?
│  ├─ Check if class exists in codebase
│  ├─ Exists? → Update import
│  └─ Removed? → Skip file with reason
├─ Wrong arguments?
│  └─ Check new signature → Update
├─ Import errors?
│  ├─ Module moved? → Update path
│  └─ Module removed? → Skip or rewrite
└─ Mock internal logic?
   └─ Rewrite to test real behavior
```

## EFFICIENCY TIPS

1. ALWAYS run ruff first - it finds problems faster than reading code
1. Batch similar errors (e.g., all F821 for same class)
1. Use ruff --fix for trivial issues, focus on real problems
1. Skip entire files if core dependency removed

## COMMON PATTERNS

**Pattern: Class removed (F821)**

```python
# Ruff says: F821 undefined name 'InputDocument'
# Check: grep -r "class InputDocument" buttermilk/
# Not found? Skip file:
import pytest
pytest.skip("InputDocument removed", allow_module_level=True)
```

**Pattern: API changed**

```python
# Ruff says: unexpected keyword argument
# OLD:
AutoGenWrapper(client=mock_client, ...)
# NEW (check actual signature):
AutoGenWrapper(client_factory=lambda: mock_client, ...)
```

**Pattern: Mock internal (F401 + unused mock)**

```python
# Ruff says: F401 'patch' imported but unused
# Code has: @patch("buttermilk.agents...")
# Fix: Remove mock, test real behavior
```

## COMMANDS REFERENCE

```bash
# Diagnostic (see all problems)
uv run ruff check tests/unit/ --output-format=concise

# Auto-fix trivial issues
uv run ruff check --fix tests/unit/

# Show specific error types
uv run ruff check tests/ --select=F821  # undefined names
uv run ruff check tests/ --select=F401  # unused imports

# Test collection only
uv run pytest tests/unit/ --co -q

# Run with stop on first fail
uv run pytest tests/unit/ -x

# Run health dashboard
uv run python scripts/test_health_dashboard.py
```

## BATCH SIZE GUIDANCE

- Fix 5-10 files per session
- Focus on one error type across multiple files
- Complete one test category before moving to next
- Track progress in test_health_report.md
