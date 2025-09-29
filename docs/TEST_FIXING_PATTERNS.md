Your job is to fix our broken tests. First read 'docs/agents/TESTER_QA.md' for general testing info.

## WORKFLOW (Follow Exactly)
1. Run health dashboard: `uv run python scripts/test_health_dashboard.py`
2. Pick category with most failures
3. **DIAGNOSTIC PHASE (Ruff)**:
   - Run: `uv run ruff check tests/[category]/ --output-format=concise`
   - This identifies ALL problems (not just fixable ones)
   - Parse output for: undefined names, wrong arguments, missing imports
   - Auto-fix trivial issues: `uv run ruff check --fix tests/[category]/`
4. **FIX PHASE** (based on ruff diagnostics):
   - Address undefined names (F821) → Check if class/function removed
   - Fix wrong arguments (E251, etc) → Update to new signatures
   - Handle import errors → Update import paths or skip if removed
5. Test collection: `uv run pytest tests/[category]/ --co -q`
6. Run tests: `uv run pytest tests/[category]/ -x`
7. Fix 5-10 files per batch
8. Commit: "fix(tests): [category] - [summary]"

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

## BATCH SIZE GUIDANCE
- Fix 5-10 files per session
- Focus on one error type across multiple files
- Complete one test category before moving to next
- Track progress in test_health_report.md

## RUFF DIAGNOSTICS GUIDE

### Priority Errors (Fix First)
| Code | Meaning | Action |
|------|---------|--------|
| F821 | Undefined name | Check if class exists → Update or skip |
| F401 | Unused import | Let ruff auto-fix |
| E999 | Syntax error | Let ruff auto-fix if possible |
| F841 | Unused variable | Let ruff auto-fix |
| B006 | Mutable default arg | Fix manually |

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

## Common Test Failure Patterns and Fixes

### 1. Syntax Errors
**Pattern**: `await` used outside async function
**Fix**: Add `async` to the function definition
```python
# Before
def test_something(self):
    result = await some_async_call()

# After  
async def test_something(self):
    result = await some_async_call()
```

**Pattern**: Extra comma in function parameters
**Fix**: Remove the extra comma
```python
# Before
async def test_function(, client):

# After
async def test_function(client):
```

### 2. Import Errors - Removed Classes
**Pattern**: Classes that no longer exist in the codebase
- `InputDocument` from `buttermilk.data.vector`
- `VectorStoreInterface` from `buttermilk.data.vector`

**Fix**: Either:
1. Skip the entire test file if it needs major refactoring:
```python
import pytest
pytest.skip("InputDocument class removed - tests need refactoring", allow_module_level=True)
```
2. Comment out the import and mark affected tests as skipped

### 3. API Contract Changes

#### AutoGenWrapper Changes
**Pattern**: `AutoGenWrapper` now requires `client_factory` instead of `client`
```python
# Before
wrapper = AutoGenWrapper(
    client=mock_client,
    model_info=model_info,
    litellm_model_name="openai/gpt-4"
)

# After
wrapper = AutoGenWrapper(
    client_factory=lambda: mock_client,
    model_info=model_info,
    litellm_model_name="openai/gpt-4"
)
```

#### FlowRunner Changes
**Pattern**: `FlowRunner.real_bm` renamed to `FlowRunner.bm`
```python
# Before
assert runner.real_bm is None

# After
assert runner.bm is None
```

#### FormattedCitation Changes
**Pattern**: Model fields changed from `text` to `title` and `citation`
```python
# Before
citation = FormattedCitation(
    text="Smith, J. (2023). Example Article.",
    style="APA"
)

# After
citation = FormattedCitation(
    title="Example Article",
    citation="Smith, J. (2023). Example Article.",
    style="APA"
)
```

### 4. Test Organization Issues

#### Collection Errors
Many test files have collection errors due to:
- Missing imports
- Undefined variables  
- Syntax errors preventing parsing

**Fix Priority (Use Ruff)**:
1. Run `uv run ruff check tests/ --output-format=concise` to identify all issues
2. Fix syntax errors first (E999) - they block everything
3. Fix undefined names (F821) - check if classes still exist
4. Fix import errors (F401) - update or remove
5. Fix API changes based on ruff diagnostics
6. Fix assertion/logic errors last

### 5. Mocking Issues - Apply New Philosophy

**Pattern**: Tests mock internal Buttermilk code
```python
# ❌ WRONG: Mocking our own code
@patch("buttermilk.agents.llm.LLMAgent._process")
def test_agent(mock_process):
    mock_process.return_value = "mocked"
```

**Fix**: Test real behavior, mock only boundaries
```python
# ✅ RIGHT: Mock only external API
@respx.mock
async def test_agent():
    respx.post("https://api.openai.com/v1/chat").mock(
        return_value=httpx.Response(200, json={"choices": [...]})
    )
    agent = LLMAgent(...)
    result = await agent._process(...)
    assert "expected" in result  # Test real behavior
```

See [TESTING_PHILOSOPHY.md](TESTING_PHILOSOPHY.md) and [testing/BOUNDARY_MOCKING.md](testing/BOUNDARY_MOCKING.md) for patterns.

### 6. Test Health Dashboard

Use the `scripts/test_health_dashboard.py` script to:
- Get an overview of all test failures
- Categorize failures by type
- Identify priority fixes (collection errors)
- Track progress across sessions

```bash
uv run python scripts/test_health_dashboard.py
```

This generates:
- `test_health_report.md` - Human-readable report
- `test_health_data.json` - Machine-readable data for tracking

## Recommended Workflow for Future Sessions

1. **Run test health dashboard** to see current state
2. **Run ruff diagnostics first** - identifies problems faster than reading code
3. **Fix collection errors first** - these prevent tests from running at all  
4. **Focus on one test category** at a time (e.g., unit/, agents/, api/)
5. **Fix 5-10 test files per session** to stay within context limits
6. **Apply testing philosophy** - remove mocks of internal code
7. **Track progress** using GitHub issues

## Progress Tracking

### Session 1 Results:
- Created test health dashboard
- Fixed 2 syntax errors (async/await issues)
- Fixed 3 import error files (marked as needing refactoring)
- Fixed AutoGenWrapper API changes (client -> client_factory)
- Fixed FlowRunner API changes (real_bm -> bm)
- Fixed FormattedCitation model changes

### Files Needing Major Refactoring:
- `tests/integration/flows/test_embeddings.py` - InputDocument removed
- `tests/integration/test_osb_multiagent_workflows.py` - VectorStoreInterface removed
- `tests/integration/test_osb_flow.py` - Multiple import issues

### Estimated Remaining Work:
- ~50-60% of tests still failing
- Most failures are in integration tests
- Unit tests improved from 62 to ~55 failures
- Need 5-10 more sessions to fully clean up