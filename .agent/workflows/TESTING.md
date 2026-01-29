---
description: 
---

## Buttermilk-Specific Requirements


- **Use real_bm fixture** - Never load config directly
- **Mock only boundaries** - Never mock buttermilk.* code
- **E2E tests required** - Task not complete without them
- **Fix systematically** - Collection errors → patterns → categories
- **Mark unclear tests** - Don't spend >5 min figuring out expected behavior

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

### CRITICAL: Live E2E Tests Required

**MANDATORY FOR TASK COMPLETION**:

All development tasks MUST include end-to-end tests that pass with:

- ✅ Real APIs (Vertex AI, Zotero, TMDB, etc.)
- ✅ Real storage (ChromaDB, BigQuery, file systems)
- ✅ Real data from actual sources
- ❌ NO mocks of Buttermilk's own code
- ❌ NO exceptions - E2E tests are non-negotiable

**A task is NOT complete until E2E tests pass.**


## Performance Testing

```bash
# Run performance tests
uv run pytest tests/performance/ -v

# Save baseline
./scripts/performance/run_performance_tests.sh --baseline

# Compare against baseline
./scripts/performance/run_performance_tests.sh --compare
```

## The real_bm Fixture

**Usage**:

```python
async def test_with_real_config(real_bm):
    """Use real Buttermilk configuration."""
    # real_bm provides:
    # - Live infrastructure (Vertex AI, ChromaDB)
    # - Real API credentials
    # - Actual configuration
    # - Structured logging

    result = await real_bm.process_flow("trans", record="test_record")
    assert result.success
```

**What real_bm provides**:

- Fully initialized execution context
- Live cloud infrastructure
- Real LLM clients (Vertex AI)
- Actual storage backends
- Complete configuration from `conf/`

**Demo Tests** (`tests/demo/`):

- Full demonstration
- Live data
- Curated output to prove functionality to human operators


## Standalone Validation - FORBIDDEN

**NEVER**:

- Create `test_*.py` files outside `tests/`
- Use `python -c "..."` for validation
- Create demo/example scripts
- Use inline Python commands for testing

**Red Flag Phrases**:

- "Let me create a test to verify..." → Use pytest in `tests/`
- "I'll run python -c to check..." → Use pytest
- "Let me validate with a quick script..." → Use pytest

