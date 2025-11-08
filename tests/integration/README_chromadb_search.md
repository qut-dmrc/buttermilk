# ChromaDB Search Tool Integration Tests

This directory contains integration tests for the ChromaDBSearchTool that demonstrate real searches against the prosocial Zotero collection.

## Prerequisites

1. **GCP Credentials**: You need valid Google Cloud credentials configured:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
   export GCP_PROJECT=your-project-id
   ```

2. **ChromaDB Collection**: The tests use the `prosocial_zot` collection configured in `conf/storage/zot.yaml`:
   - Collection name: `prosocial_zot`
   - Embedding model: `gemini-embedding-001`
   - Storage location: `gs://prosocial-dev/data/zotero-prosocial-fulltext/files`

## Test Files

### `test_chromadb_search_tool_integration.py`

Full pytest integration test suite that covers:

- Basic search functionality ("what are transaction costs")
- Filtered searches by metadata
- No-duplicate search mode
- Error handling
- Collection statistics
- Tool function interface for agents

### `demo_chromadb_search.py`

Standalone demo script that shows:

- Multiple search queries
- Result formatting and metadata
- Filtered searches
- No-duplicate mode

## Running the Tests

### Run the pytest suite:

```bash
# From the project root
cd /src/buttermilk
uv run pytest tests/integration/test_chromadb_search_tool_integration.py -v -s

# Run a specific test
uv run pytest tests/integration/test_chromadb_search_tool_integration.py::TestChromaDBSearchToolIntegration::test_search_transaction_costs -v -s
```

### Run the demo script:

```bash
# From the project root
cd /src/buttermilk
uv run python tests/integration/demo_chromadb_search.py
```

## Example Output

When running the demo, you'll see output like:

```
=== ChromaDB Search Tool Demo ===
Using collection: prosocial_zot
Embedding model: gemini-embedding-001

Collection contains 12345 embeddings

============================================================
QUERY: what are transaction costs
============================================================

--- Result 1 ---
Document: Transaction Cost Economics and Governance
Document ID: ABC123
Chunk ID: ABC123_5
Score: 0.8532
Content: Transaction costs are the costs of making an economic exchange. They include search and information costs, bargaining costs, and policing and enforcement costs...
Metadata: {'chunk_index': 5, 'content_type': 'abstract', 'embedding_model': 'gemini-embedding-001'}
```

## Understanding the Results

Each search result includes:

- **Document Title**: The title of the source document
- **Document ID**: The Zotero key of the document
- **Chunk ID**: Unique identifier for this specific text chunk
- **Score**: Similarity score (higher is better, max 1.0)
- **Content**: The actual text that matched your query
- **Metadata**: Additional information about the chunk

## Search Tips

1. **Use natural language queries**: "what are transaction costs" works better than just "transaction costs"
2. **Filter by content type**: Use `where={"content_type": "abstract"}` to search only abstracts
3. **Adjust result count**: Use `n_results` parameter (default is 10)

## Troubleshooting

If tests fail:

1. Check GCP credentials are properly configured
2. Verify you have access to the GCS bucket
3. Ensure the ChromaDB collection exists and has been populated
4. Check network connectivity to Google Cloud

## Integration with Agents

The ChromaDBSearchTool can be used by agents via its `as_tool()` method:

```python
search_tool = ChromaDBSearchTool(**config)
await search_tool.initialize()
function_tool = search_tool.as_tool()

# Use in agent
result = await function_tool.run_json({"query": "your search query"})
```
