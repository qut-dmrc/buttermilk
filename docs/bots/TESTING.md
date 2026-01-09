# Buttermilk Testing Instructions

**Load framework TESTING.md for generic testing philosophy** (`@$ACADEMICOPS/core/TESTING.md`).

This file contains Buttermilk-specific testing requirements.

## CRITICAL: Live E2E Tests Required

**MANDATORY FOR TASK COMPLETION**:

All development tasks MUST include end-to-end tests that pass with:

- ✅ Real APIs (Vertex AI, Zotero, TMDB, etc.)
- ✅ Real storage (ChromaDB, BigQuery, file systems)
- ✅ Real data from actual sources
- ❌ NO mocks of Buttermilk's own code
- ❌ NO exceptions - E2E tests are non-negotiable

**A task is NOT complete until E2E tests pass.**

## True End-to-End Testing

**Definition**: TRUE E2E = Real APIs + Real Storage + Real Data. NO MOCKS.

**Pattern**:

```python
@pytest.mark.anyio
async def test_zotero_to_chromadb_pipeline(real_bm):
    """TRUE E2E test - REAL everything."""

    # REAL Zotero API fetch
    zotero_source = ZoteroSource(library_id=config.zotero.library_id)
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

**Key principles**:

- Use `real_bm` fixture for configuration
- Call REAL APIs (Vertex AI actually makes API calls)
- Store in REAL databases (temp locations OK for isolation)
- NO mocks of Buttermilk code
- Validate end-to-end behavior

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

## Test Categories

**Unit Tests** (`tests/unit/`):

- Individual functions/classes
- Mock external APIs only (Vertex AI, Zotero API)
- Never mock Buttermilk's own code
- Fast, isolated

**Integration Tests** (`tests/integration/`):

- Multiple Buttermilk components together
- Use `real_bm` fixture
- May mock external APIs if needed
- Test component integration

**End-to-End Tests** (`tests/endtoend/`):

- MANDATORY for task completion
- Complete workflows (fetch → process → store)
- REAL APIs, REAL storage, REAL data
- NO mocks except truly unavoidable external systems
- Run with `pytest -m endtoend`

**Demo Tests** (`tests/demo/`):

- Full demonstration
- Live data
- Curated output to prove functionality to human operators

## When to Mock (Rare)

**Mock ONLY**:

- External APIs you don't control (and can't use test credentials for)
- Third-party services (document why)

**NEVER Mock**:

- ❌ Buttermilk's own modules (`buttermilk.*`)
- ❌ `SemanticSplitter`, `EmbeddingGenerator`, etc.
- ❌ `ChromaDBEmbeddings`, `BigQueryStorage`
- ❌ `ExecutionContext`, configuration
- ❌ Flow orchestration

**Example - What NOT to mock**:

```python
# ❌ WRONG - mocking our own code
@patch("buttermilk.processors.embeddings.EmbeddingGenerator")
@patch("buttermilk.data.vector.ChromaDBEmbeddings")
async def test_pipeline(mock_embeddings, mock_chromadb):
    # This is NOT E2E - it's all mocks!
    pass


# ✅ CORRECT - use real components
async def test_pipeline(real_bm):
    embedder = EmbeddingGenerator(...)  # Real
    storage = ChromaDBEmbeddings(...)  # Real
    # Real pipeline execution
```

## Test Dependencies

**Install test dependencies**:

```bash
uv sync --extra dev --extra research --extra azure --upgrade
```

**Required for E2E tests**:

- Vertex AI credentials (service account JSON)
- Test API keys in `.env` (Zotero, TMDB)
- ChromaDB dependencies
- All processor dependencies

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

**Tester Subagent Available**:

```
Task: tester - [describe testing need]
```

Use tester subagent for ALL testing scenarios.

## Test Isolation

**Temporary resources**:

```python
# Temp directory for ChromaDB
with tempfile.TemporaryDirectory() as tmpdir:
    storage = ChromaDBEmbeddings(persist_directory=tmpdir)

# Unique collection names
collection = f"test_{uuid.uuid4()}"
```

## Success Criteria

**E2E tests must**:

1. ✅ Use `real_bm` fixture
2. ✅ Call real APIs (document any mocked external systems)
3. ✅ Store in real databases (temp locations OK)
4. ✅ Exercise complete workflow
5. ✅ Validate end-to-end behavior
6. ✅ Use realistic test data
7. ✅ Clean up resources
8. ✅ PASS before task is considered complete

**Remember**: NO task is complete without passing E2E tests on real data with real APIs.

See framework TESTING.md for generic testing philosophy and pytest patterns.

See `bots/docs/_CHUNKS/E2E-TESTING.md` for detailed E2E testing examples.
