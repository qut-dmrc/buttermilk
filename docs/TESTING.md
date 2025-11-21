# Buttermilk Testing Guide

## Core Philosophy: Mock Only at System Boundaries

We mock only at system boundaries—where our code interfaces with external systems. The "inside" of our application is tested with real logic and assertions.

**System Boundaries (OK to Mock)**:
- Network: HTTP calls, API requests, websockets
- Filesystem: File I/O (use `tmp_path`)
- Time: System clock (use `freezegun`)
- Environment: Environment variables

**Our Code (NEVER Mock)**:
- Anything in `buttermilk.*`
- Business logic, data transformations, agents
- Internal APIs and orchestration

**Why?** Mocking internal code tests the mock, not your logic. Mock only external dependencies.

```python
# ❌ BAD: Testing the mock
@patch("buttermilk.agents.llm.LLMAgent._process")
def test_agent(mock_process):
    mock_process.return_value = "mocked"
    assert agent.run() == "mocked"  # Proves nothing!

# ✅ GOOD: Test real behavior, mock external API
@respx.mock
def test_agent():
    respx.post("https://api.openai.com/v1/chat").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Paris"}}]})
    )
    result = agent.answer("Capital of France?")
    assert "Paris" in result
```

## Buttermilk-Specific Requirements

### Critical: Use real_bm Fixture

**ALL Buttermilk tests MUST use `real_bm` or `real_conf` fixtures.**

```python
@pytest.mark.anyio
async def test_with_real_config(real_bm):
    """Test using real Buttermilk infrastructure."""
    # real_bm provides:
    # - Live Vertex AI, ChromaDB
    # - Real API credentials
    # - Configuration from conf/
    # - Structured logging

    result = await real_bm.process_flow("trans", record="test_record")
    assert result.success
```

**❌ FORBIDDEN: Direct config loading**

```python
# ❌ DON'T DO THIS
from hydra import initialize_config_dir, compose

def test_something():
    with initialize_config_dir(config_dir="conf/"):
        cfg = compose(config_name="config")  # Bypasses test infrastructure!

# ✅ DO THIS INSTEAD
def test_something(real_bm):
    # Configuration already loaded properly
    assert real_bm.config.model_name == "gemini-1.5-pro"
```

### True End-to-End Testing

**TRUE E2E = Real APIs + Real Storage + Real Data. NO MOCKS of Buttermilk code.**

```python
@pytest.mark.anyio
async def test_complete_pipeline(real_bm):
    """TRUE E2E - uses REAL everything."""

    # REAL Zotero API fetch
    records = await ZoteroSource(library_id=real_bm.config.zotero.library_id).fetch(limit=1)

    # REAL processing
    chunks = [chunk async for chunk in SemanticSplitter(chunk_size=100).process(records[0])]
    embedded = [emb async for emb in EmbeddingGenerator(model="text-embedding-004").process(chunks[0])]

    # REAL ChromaDB storage (temp location)
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ChromaDBEmbeddings(persist_directory=tmpdir, collection_name="e2e_test")
        result = await storage.process_record(embedded[0])

    assert result.status == "processed"
```

**Task completion requirement**: E2E tests MUST pass before task is considered complete.

## Test Structure

```
tests/
├── unit/           # Isolated components
├── integration/    # Components working together
├── endtoend/       # Complete workflows (real everything)
├── api/            # API-specific tests
└── groupchat/      # Multi-agent interactions
```

Configuration: `conf/testing.yaml`

## Practical Testing Patterns

### Network Boundaries
```python
@respx.mock
async def test_api_call():
    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    result = await fetch_external_data()
    assert result["status"] == "ok"
```

### Filesystem Boundaries
```python
def test_file_processing(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    result = process_file(test_file)
    assert result == "PROCESSED: test content"
```

### Time Boundaries
```python
from freezegun import freeze_time

@freeze_time("2024-01-01")
def test_timestamp():
    record = create_record()
    assert record.timestamp == "2024-01-01T00:00:00"
```

### Simple Test Doubles
```python
class FakeLLM:
    """Simple test double for LLM interactions."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def generate(self, prompt):
        self.calls.append(prompt)
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        return "default response"

async def test_agent_with_fake():
    fake_llm = FakeLLM(responses={"capital of France": "Paris"})
    agent = Agent(llm=fake_llm)
    result = await agent.research("Tell me about Paris")
    assert "Paris" in result
```

## Test Fixing Workflow

### Phase 1: Collection Errors (Highest Priority)
```bash
# Fix import issues and syntax errors first
uv run ruff check tests/ --output-format=concise | grep "F821\|import"
uv run ruff check --fix tests/

# These block test execution - fix immediately
```

### Phase 2: Pattern-Based Fixes
Look for patterns across multiple tests:
- Field renames (e.g., `name` → `project_name`)
- Type changes (e.g., `StorageConfig` → `BigQueryStorageConfig`)
- Import paths (classes moved to new modules)

```bash
# Find pattern
grep -r "old_pattern" tests/ --include="*.py"

# Fix in bulk if straightforward
find tests/ -name "*.py" -exec sed -i 's/old_pattern/new_pattern/g' {} \;
```

### Phase 3: Category-Specific Fixes
Pick ONE category, fix systematically:
```bash
uv run ruff check tests/unit/ --output-format=concise
uv run pytest tests/unit/ -v --tb=no
# Fix issues, commit, move to next category
```

**Category order**: unit → integration → api → groupchat → endtoend

### Phase 4: Mark Tests Needing Decisions
When expected behavior is unclear (>5 min to figure out):

```python
@pytest.mark.skip(reason="NEEDS DECISION: Test expects X but code now does Y. Question: Which is correct?")
def test_something(): ...
```

Document in GitHub issue, don't create tracking files.

## Decision Tree for Failing Tests

```
Test fails
├─ Collection error? → Fix import/syntax (highest priority)
├─ Ruff auto-fixable? → Run ruff --fix
├─ Quick fix obvious (<2 min)? → Fix immediately
│  ├─ Field rename? → Update
│  ├─ Import path? → Update
│  └─ Expected value clearly changed? → Update
├─ Needs investigation (>5 min)? → Mark for decision, move on
```

## Red Flags in Tests

**Signs a test needs refactoring**:
1. Mocking our own code: `@patch("buttermilk._core.something")`
2. Complex mock setup: `mock.method.return_value.attribute.side_effect = ...`
3. Testing mock behavior: `mock.assert_called_with(...)`
4. Mocking data transformations instead of testing logic

## Good Test Checklist

✅ Tests real code execution paths
✅ Mocks only external system boundaries
✅ Uses simple test doubles over complex mocks
✅ Assertions verify actual behavior
✅ Tests remain valid when implementation changes
✅ Readable and maintainable

## Common Ruff Diagnostics

| Code | Meaning         | Action                        | Priority |
|------|-----------------|-------------------------------|----------|
| E999 | Syntax error    | Auto-fix or manual fix        | Critical |
| F821 | Undefined name  | Check if exists → fix or skip | High     |
| F401 | Unused import   | Auto-fix                      | Medium   |
| F841 | Unused variable | Auto-fix                      | Low      |

## Test Dependencies

```bash
# Install all test dependencies
uv sync --extra dev --extra research --extra azure --upgrade
```

**Required for E2E tests**:
- Vertex AI credentials (service account JSON)
- Test API keys in `.env` (Zotero, TMDB)
- ChromaDB dependencies

## Performance Testing

```bash
# Run performance tests
uv run pytest tests/performance/ -v

# Save baseline
./scripts/performance/run_performance_tests.sh --baseline

# Compare against baseline
./scripts/performance/run_performance_tests.sh --compare
```

## Commit Strategy

**Commit frequently** at natural breakpoints:
- After fixing all collection errors
- After bulk pattern replacement
- After fixing a category (<50 changes)
- After fixing specific error pattern

**Commit message format**:
```
fix(tests): [scope] - [concise summary]

- Fixed [specific thing 1]
- Updated [specific thing 2]

Pass rate: X% → Y%
```

## Summary

- **Use real_bm fixture** - Never load config directly
- **Mock only boundaries** - Never mock buttermilk.* code
- **E2E tests required** - Task not complete without them
- **Fix systematically** - Collection errors → patterns → categories
- **Mark unclear tests** - Don't spend >5 min figuring out expected behavior
