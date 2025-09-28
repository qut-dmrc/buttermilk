# Testing Enforcement Guidelines

## 🚨 MANDATORY: Testing Anti-Pattern Detection

This document supplements the development workflow with specific guidance to prevent testing laziness and shortcuts.

## Immediate Red Flags - STOP CODING If You Catch Yourself:

### 1. Creating Test Files Outside `tests/` Directory
```bash
# 🚫 NEVER create these files:
test_anything.py                    # Root directory
src/test_something.py               # Source directory
buttermilk/test_feature.py          # Module directory
scripts/test_validation.py          # Scripts directory

# ✅ ALWAYS create tests here:
tests/test_feature.py               # Root tests
tests/agents/test_agent_name.py     # Agent tests
tests/flows/test_flow_name.py       # Flow tests
```

### 2. Using Wrong Async Markers
The project uses `anyio`, not `asyncio`:
```python
# 🚫 WRONG:
@pytest.mark.asyncio
async def test_something():

# ✅ CORRECT:
@pytest.mark.anyio
async def test_something(real_bm):  # Use fixtures!
```

### 3. Ignoring Fixtures
The `conftest.py` provides essential fixtures - USE THEM:
```python
# 🚫 WRONG: Standalone initialization
async def test_something():
    # Manual setup without BM instance

# ✅ CORRECT: Use provided fixtures
async def test_something(real_bm, real_conf):
    # Proper initialization via fixtures
```

### 4. Creating "Validation Scripts"
```python
# 🚫 NEVER do this:
def validate_my_implementation():
    # Test logic here
    print("Testing...")

if __name__ == "__main__":
    validate_my_implementation()

# ✅ ALWAYS do this:
@pytest.mark.anyio
async def test_implementation_works():
    """Proper test description."""
    # Test logic with assertions
    assert expected_behavior()
```

## Enforcement Checklist

Before creating ANY test file, verify:

- [ ] File is in `tests/` directory
- [ ] Uses `@pytest.mark.anyio` for async tests
- [ ] Imports proper fixtures from conftest.py
- [ ] Uses `real_bm` fixture when BM instance needed
- [ ] Has proper test function names (`test_*`)
- [ ] Uses assertions, not print statements
- [ ] Can be run with `uv run pytest`

## Common Excuses and Responses

### "I'm just doing a quick validation"
**Response**: No such thing. Use pytest or don't test.

### "This is just a proof of concept"
**Response**: Proof of concepts need proper tests more than production code.

### "I'll move it to tests/ later"
**Response**: Do it now. "Later" never comes.

### "It's easier to debug with print statements"
**Response**: Use pytest's `-s` flag: `uv run pytest -s tests/test_file.py`

## Testing Workflow Enforcement

### Step 1: Create Failing Test FIRST
```python
# tests/test_new_feature.py
@pytest.mark.anyio
async def test_feature_behavior(real_bm):
    """Test demonstrating the expected behavior."""
    # This should fail until feature is implemented
    result = await new_feature.process(input_data)
    assert result.status == "expected"
```

### Step 2: Run Test to Confirm It Fails
```bash
uv run pytest tests/test_new_feature.py::test_feature_behavior -v
# Should fail with clear error message
```

### Step 3: Implement Minimal Fix
Only implement what's needed to make the test pass.

### Step 4: Run Test to Confirm It Passes
```bash
uv run pytest tests/test_new_feature.py::test_feature_behavior -v
# Should now pass
```

### Step 5: Run Full Test Suite
```bash
uv run pytest tests/
# Ensure no regressions
```

## Fixture Usage Patterns

### BM Instance Required
```python
@pytest.mark.anyio
async def test_needs_bm_instance(real_bm):
    """Test requiring full BM initialization."""
    # real_bm provides configured BM instance
    storage = real_bm.get_storage(config)
```

### Configuration Required
```python
@pytest.mark.anyio
async def test_needs_config(real_bm, real_conf):
    """Test requiring configuration access."""
    # real_conf provides resolved configuration
    flow_config = real_conf.flows.my_flow
```

### LLM Testing
```python
@pytest.mark.anyio
async def test_llm_behavior(real_llm):
    """Test with real LLM instance."""
    # real_llm provides configured LLM
    response = await real_llm.process(prompt)
```

## Detection Commands

Run these to catch testing anti-patterns:

```bash
# Find test files outside tests/ directory
find . -name "test_*.py" -not -path "./tests/*" -not -path "./.venv/*"

# Find standalone test scripts
grep -r "if __name__.*main" --include="test_*.py" .

# Find wrong async markers
grep -r "@pytest.mark.asyncio" tests/

# Find tests missing fixtures
grep -A 5 "async def test_" tests/ | grep -B 5 -A 5 "real_bm\|real_conf"
```

## Consequences of Violations

1. **Immediate**: Test won't run properly (fixtures missing, wrong markers)
2. **Medium-term**: Time wasted debugging test infrastructure instead of actual code
3. **Long-term**: Technical debt, unreliable tests, broken CI/CD

## Remember

Testing is not optional or negotiable. If you're not using pytest with proper fixtures and structure, you're not testing - you're just pretending to test.

**When in doubt, look at existing tests in `tests/` and follow their patterns exactly.**