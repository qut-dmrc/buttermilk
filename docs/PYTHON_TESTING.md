# Buttermilk Python Testing Standards

**Project-specific testing requirements for Buttermilk.** For generic testing philosophy, see `@$AOPS/skills/python-dev/references/testing.md`.

## Buttermilk-Specific Requirements

### Critical: Use real_bm Fixture

**ALL Buttermilk tests MUST use the `real_bm` or `real_conf` fixtures from conftest.py.**

```python
import pytest


@pytest.mark.anyio
async def test_with_real_config(real_bm):
    """Test using real Buttermilk configuration."""

    # real_bm provides:
    # - Live infrastructure (Vertex AI, ChromaDB)
    # - Real API credentials
    # - Actual configuration from conf/
    # - Structured logging

    result = await real_bm.process_flow("trans", record="test_record")
    assert result.success
```

**What real_bm provides**:

- Fully initialized execution context
- Live cloud infrastructure
- Real LLM clients (Vertex AI)
- Actual storage backends (ChromaDB, BigQuery)
- Complete configuration loaded via Hydra

### FORBIDDEN: Direct Config Loading

**❌ NEVER do this in tests**:

```python
# ❌ FORBIDDEN - loads config directly
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture
def config():  # DON'T create fixtures that load Hydra
    with initialize_config_dir(config_dir="conf/"):
        cfg = compose(config_name="config")
        yield cfg


# ❌ FORBIDDEN - bypasses test infrastructure
def test_something():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir="conf/"):
        cfg = compose(config_name="config")
        # ...
```

**✅ Use real_bm or real_conf instead**:

```python
@pytest.mark.anyio
async def test_something(real_bm):
    """Use real_bm fixture instead."""
    # Configuration already loaded properly
    assert real_bm.config.model_name == "gemini-1.5-pro"
```

## True End-to-End Testing

**TRUE E2E = Real APIs + Real Storage + Real Data. NO MOCKS of Buttermilk code.**

```python
@pytest.mark.anyio
async def test_zotero_to_chromadb_pipeline(real_bm):
    """TRUE E2E test - uses REAL everything."""

    # REAL Zotero API fetch
    zotero_source = ZoteroSource(library_id=real_bm.config.zotero.library_id)
    records = await zotero_source.fetch(limit=1)

    # REAL semantic splitter
    splitter = SemanticSplitter(chunk_size=100)
    chunks = [chunk async for chunk in splitter.process(records[0])]

    # REAL Vertex AI embeddings
    embedder = EmbeddingGenerator(model="text-embedding-004")
    embedded = [emb async for emb in embedder.process(chunks[0])]

    # REAL ChromaDB storage
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ChromaDBEmbeddings(
            persist_directory=str(Path(tmpdir) / "test_db"),
            collection_name="e2e_test",
        )
        result = await storage.process_record(embedded[0])

    # Verify complete pipeline
    assert result.status == "processed"
    assert result.chunks_created > 0
```

## Test Structure

**Test categories**:

- `tests/unit/` - Isolated component tests
- `tests/integration/` - Components working together
- `tests/endtoend/` - Complete workflows with real everything
- `tests/api/` - API-specific tests
- `tests/groupchat/` - Multi-agent interaction tests

**All test configuration**: `conf/testing.yaml`

## Mandatory for Task Completion

**A task is NOT complete until E2E tests pass.**

Every development task in Buttermilk MUST include end-to-end tests that:

1. ✅ Use `real_bm` fixture for configuration
2. ✅ Call real APIs (Vertex AI, Zotero, TMDB, etc.)
3. ✅ Store in real databases (ChromaDB, BigQuery) in temp locations
4. ✅ Exercise complete workflow from input to output
5. ✅ Validate end-to-end behavior
6. ✅ Use realistic test data
7. ✅ Clean up resources after test
8. ✅ **PASS before task is considered complete**

## What NOT to Mock

**❌ NEVER mock Buttermilk's own code**:

- `buttermilk.*` modules
- `SemanticSplitter`, `EmbeddingGenerator`, etc.
- `ChromaDBEmbeddings`, `BigQueryStorage`
- `ExecutionContext`, configuration
- Flow orchestration

**✅ Only mock external boundaries** (and only when necessary):

- External third-party APIs you can't call
- Services requiring paid credentials for every test

## Test Dependencies

```bash
# Install test dependencies
uv sync --extra dev --extra research --extra azure --upgrade
```

**Required for E2E tests**:

- Vertex AI credentials (service account JSON)
- Test API keys in `.env` (Zotero, TMDB)
- ChromaDB dependencies
- All processor dependencies

## Performance Testing

Buttermilk uses `pytest-benchmark` for performance testing.

```bash
# Run performance tests
uv run pytest tests/performance/ -v

# Save baseline
./scripts/performance/run_performance_tests.sh --baseline

# Compare against baseline
./scripts/performance/run_performance_tests.sh --compare
```

**Performance goals**:

- ChromaDB lazy initialization: < 1s
- MCP server startup: < 30s
- Background warmup completes after configured delay

See `/home/nic/src/buttermilk/docs/PERFORMANCE_TESTING.md` for full guide.

## Test Fixing Workflow

When fixing broken tests, follow this cycle-based approach:

### Phase 1: Collection Errors (Highest Priority)

```bash
# Fix import issues and syntax errors
uv run ruff check tests/ --output-format=concise | grep "F821\|import"
uv run ruff check --fix tests/
```

### Phase 2: Category-Specific Fixes

Pick ONE category, fix systematically, commit:

```bash
# Run ruff on category
uv run ruff check tests/unit/ --output-format=concise

# Run tests
uv run pytest tests/unit/ -v --tb=no

# Fix 5-10 files
# Commit: fix(tests): unit - [summary]
```

**Category order** (by ease):

1. `unit/` - Usually simpler
2. `integration/` - Medium complexity
3. `api/` - Depends on setup
4. `groupchat/` - Complex interactions
5. `endtoend/` - May need external services

### Phase 3: Commit Frequently

Commit at natural breakpoints:

- After fixing collection errors
- After bulk pattern replacement
- After fixing a category (if <50 changes)
- After fixing specific error pattern

**Never**:

- Massive commits with 100+ files
- Mixing unrelated fixes
- Committing broken tests

## Common Buttermilk Patterns

### Using real_logger

```python
def test_with_logging(real_bm, real_logger):
    """Test with real logger."""
    real_logger.info("Starting test")
    # Test code
```

### Using real_llm

```python
@pytest.mark.anyio
async def test_with_llm(real_llm):
    """Test with real LLM client."""
    response = await real_llm.generate("Test prompt")
    assert response.text
```

### Async Tests

```python
@pytest.mark.anyio
async def test_async_operation(real_bm):
    """All async tests need @pytest.mark.anyio."""
    result = await async_function()
    assert result
```

## Summary

- **Use real_bm fixture** - Never load config directly
- **E2E tests required** - Task not complete without them
- **Real APIs, real storage** - No mocking Buttermilk code
- **Test categories** - unit/, integration/, endtoend/
- **Performance testing** - pytest-benchmark for optimization
- **Fix systematically** - Cycle-based approach for test fixes
