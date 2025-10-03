# Zotero Vector Database Guide

This guide explains how to build a ChromaDB vector database from your Zotero library using Buttermilk.

## Quick Start

### 1. Set up credentials
```bash
export ZOTERO_API_KEY="your-zotero-api-key"
export ZOTERO_LIBRARY_ID="your-library-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcs-credentials.json"  # If using GCS
```

### 2. Build the vector database
```bash
# Using the enhanced Zotero configuration with Google Gemini embeddings
python -m buttermilk.data.vector

# The default configuration uses conf/run/vectorise.yaml
# To use the enhanced Zotero-specific configuration, modify conf/config.yaml
# to set: run: vectorise_zotero
```

## Important Notes

- The system now uses Google Gemini embeddings (`gemini-embedding-001`) instead of Vertex AI
- Ensure you have sufficient quota for embedding API calls
- The enhanced features include progress tracking, graceful interruption, and batch synchronization

## Configuration Options

The main configuration file is `conf/run/vectorise_zotero.yaml`. You can override any setting from the command line:

### Limit documents (for testing)
```bash
python -m buttermilk.data.vector run=vectorise_zotero max_docs=50
```

### Resume from a specific offset
```bash
python -m buttermilk.data.vector run=vectorise_zotero start_from=1000
```

### Use different storage location
```bash
python -m buttermilk.data.vector run=vectorise_zotero vectoriser.persist_directory=/path/to/chromadb
```

### Adjust batch processing
```bash
python -m buttermilk.data.vector run=vectorise_zotero \
  vectoriser.sync_batch_size=100 \
  vectoriser.concurrency=10
```

### Quiet mode (no progress bar)
```bash
python -m buttermilk.data.vector run=vectorise_zotero quiet=true
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
- Automatically skips documents already in the vector store
- Multiple strategies: by record ID, content hash, or both

### Remote Storage Support
- Automatic sync to Google Cloud Storage
- Smart caching for remote ChromaDB
- Configurable sync intervals

## Configuration Files

### Main Configuration: `conf/run/vectorise_zotero.yaml`
```yaml
name: zotero_vectorizer
job: zotero

# Vector store configuration
vectoriser:
  _target_: buttermilk.data.vector.ChromaDBEmbeddings
  persist_directory: ${storage.persist_directory}
  collection_name: ${storage.collection_name}
  embedding_model: ${storage.embedding_model}
  dimensionality: ${storage.dimensionality}
  concurrency: 20
  sync_batch_size: 50
  deduplication_strategy: both
  
# Chunking configuration  
chunker:
  _target_: buttermilk.data.vector.SemanticSplitter
  chunk_size: 4000
  chunk_overlap: 1000

# Zotero source
input_docs:
  _target_: buttermilk.libs.zotero.ZotDownloader
  library: ${oc.env:ZOTERO_LIBRARY_ID}
  save_dir: ${oc.env:HOME}/.cache/buttermilk/zotero/pdfs
```

### Storage Configuration: `conf/storage/zot.yaml`
```yaml
type: chromadb
persist_directory: "gs://your-bucket/chromadb"
collection_name: "zotero_collection"
embedding_model: "text-embedding-005"
dimensionality: 768
```

## Monitoring Progress

The enhanced pipeline provides detailed progress information:

1. **Startup Banner**: Shows configuration details
2. **Progress Bar**: Real-time updates with documents processed
3. **Summary Statistics**: 
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