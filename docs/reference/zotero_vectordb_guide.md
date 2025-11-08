# Zotero Vector Database Guide

This guide explains how to build a ChromaDB vector database from your Zotero library using Buttermilk.

**Note**: Application-specific configurations (like zotmcp, osbchatmcp) should live in their respective project directories, not in buttermilk. Buttermilk provides the library components; applications provide the configs.

## Quick Start

### 1. Set up credentials

```bash
export ZOTERO_API_KEY="your-zotero-api-key"
export ZOTERO_LIBRARY_ID="your-library-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcs-credentials.json"  # If using GCS
```

### 2. Build the vector database

```bash
# Run from your application directory (e.g., zotmcp)
cd /path/to/your/app

# Using a simple runner script (recommended)
uv run python scripts/run_vectorization.py

# Or with a custom script using Hydra
python your_pipeline_runner.py
```

See `projects/zotmcp/scripts/run_vectorization.py` for a simple runner script example, and `projects/zotmcp/conf/vectorize.yaml` for the config.

## Important Notes

- The system now uses Google Gemini embeddings (`gemini-embedding-001`) instead of Vertex AI
- Ensure you have sufficient quota for embedding API calls
- The enhanced features include progress tracking, graceful interruption, and batch synchronization

## Configuration Options

You can override any setting from the command line when running from your application directory:

### Limit documents (for testing)

```bash
uv run python scripts/run_vectorization.py run.limit=50
```

### Adjust batch processing

```bash
uv run python scripts/run_vectorization.py \
  vectoriser.sync_batch_size=100 \
  vectoriser.concurrency=10
```

### Change deduplication strategy

```bash
uv run python scripts/run_vectorization.py \
  vectoriser.deduplication_strategy=both
```

## Features

### Progress Tracking

- Real-time progress bar showing documents processed
- Detailed statistics including processing rate
- Summary report at completion

### Graceful Interruption

- Press Ctrl+C to stop processing after current batch
- Automatically resumes where it left off on next run

### Multi-field Embeddings

The system creates separate embeddings for:

- **Full text**: Main document content (chunked)
- **Abstract**: Paper abstract (if available)
- **Annotations**: PDF annotations (when implemented)
- **Tags**: Zotero tags
- **Notes**: Zotero notes

### Deduplication

Deduplication happens **early** by configuring `ZotDownloader` with a `vector_store` reference:

```yaml
input_docs:
  _target_: buttermilk.libs.zotero.ZotDownloader
  vector_store: ${vectoriser} # Explicit reference to ChromaDB
```

**How It Works**:

1. **Source-level checking**: `ZotDownloader` queries ChromaDB **before** downloading PDFs
1. **Early exit**: If record exists, `ZotDownloader` doesn't yield it to the pipeline
1. **Cost savings**: Skips expensive operations for existing records:
   - ❌ No PDF download from Zotero API
   - ❌ No text extraction from PDF
   - ❌ No chunking
   - ❌ No re-embedding

**Deduplication Strategies** (configured via `deduplication_strategy` on `ChromaDBEmbeddings`):

- `"record_id"`: Skip if Zotero item ID exists in ChromaDB (fastest, ideal for incremental sync)
- `"content_hash"`: Skip if content hash matches (detects when documents are modified)
- `"both"`: Skip only if BOTH record_id AND content_hash match (most thorough, re-processes modified documents)

**Example**: With 6GB of existing embeddings and no pipeline cache:

- Set `deduplication_strategy="record_id"` on ChromaDBEmbeddings
- Set `vector_store: ${vectoriser}` on ZotDownloader
- Only net-new Zotero records will be downloaded and vectorized

### Remote Storage Support

- Automatic sync to Google Cloud Storage
- Smart caching for remote ChromaDB
- Configurable sync intervals

## Configuration Files

### Example Configuration: `yourapp/conf/vectorize.yaml`

This example shows the complete structure. See `projects/zotmcp/conf/vectorize.yaml` for a working implementation.

```yaml
# @package _global_

defaults:
  - base_config # Your app's base config
  - _self_

vectoriser:
  _target_: buttermilk.data.vector.ChromaDBEmbeddings
  persist_directory: "gs://your-bucket/zotero/chromadb"
  collection_name: "zotero_library"
  embedding_model: "gemini-embedding-001"
  dimensionality: 3072
  concurrency: 10
  sync_batch_size: 50
  deduplication_strategy: record_id # Fast deduplication for incremental sync
  enable_record_cache: true

pipeline:
  _target_: buttermilk.pipeline.PipelineOrchestrator
  pipeline_name: zotero_vectorization
  concurrency: 5
  limit: null # Process all (or use run.limit from CLI)

  # Zotero source with deduplication
  source:
    _target_: buttermilk.libs.zotero.ZotDownloader
    library: ${oc.env:ZOTERO_LIBRARY_ID}
    save_dir: .cache/zotero/items
    download_concurrency: 8
    vector_store: ${vectoriser} # Enable early deduplication

  processors:
    # 1. Chunk text
    - _target_: buttermilk.data.vector.SemanticSplitter
      chunk_size: 1000
      chunk_overlap: 250

    # 2. Embed and upload
    - ${vectoriser}
```

**Key Configuration**: `vector_store: ${vectoriser}` connects ZotDownloader to ChromaDB, enabling it to skip downloading PDFs for records that already exist.

## Monitoring Progress

The enhanced pipeline provides detailed progress information:

1. **Startup Banner**: Shows configuration details
1. **Progress Bar**: Real-time updates with documents processed
1. **Summary Statistics**:
   - Total documents found
   - Successfully processed
   - Failed documents
   - Processing rate
   - Total embeddings in collection

## Troubleshooting

### Memory Issues

Reduce batch size and concurrency:

```bash
python -m buttermilk.data.vector run=vectorise_zotero \
  vectoriser.sync_batch_size=25 \
  vectoriser.concurrency=5
```

### API Rate Limits

Add delays between batches:

```bash
python -m buttermilk.data.vector run=vectorise_zotero \
  vectoriser.embedding_cooldown_seconds=1.0
```

### Debug Mode

Enable debug logging:

```bash
export BUTTERMILK_LOG_LEVEL=DEBUG
python -m buttermilk.data.vector run=vectorise_zotero
```

## Using the Vector Database

Once built, use the vector database with RAG agents:

```yaml
# conf/flows/zotero_rag.yaml
defaults:
  - /storage: zot
  - /agents@agents.researcher: rag_zot

orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
```

Then start the chat:

```bash
python -m buttermilk.runner.cli +flow=zotero_rag run=api
```
