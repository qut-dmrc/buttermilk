# OSB Vector Database Guide

This guide provides comprehensive documentation for working with the Online Safety Bureau (OSB) dataset using Buttermilk's vector database infrastructure.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Reference](#configuration-reference)
4. [Data Loading](#data-loading)
5. [Vector Store Setup](#vector-store-setup)
6. [Search and Retrieval](#search-and-retrieval)
7. [Integration with Buttermilk Flows](#integration-with-buttermilk-flows)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Overview

The OSB (Online Safety Bureau) dataset contains legal case summaries and decisions related to online safety, content moderation, and platform responsibility. Buttermilk provides a complete infrastructure for:

- Loading OSB JSON data from cloud storage
- Creating searchable vector databases using ChromaDB
- Performing semantic search over legal documents
- Integrating with LLM agents for legal analysis

### Key Features

- **Pre-configured OSB flow** with optimal settings for legal text
- **Cloud storage integration** with automatic caching
- **Vertex AI embeddings** for high-quality semantic search
- **Specialized prompts** for legal analysis
- **Scalable async processing** for large datasets
- **Metadata preservation** for rich filtering

## Quick Start

### 1. Basic Setup

```python
from buttermilk.utils.nb import init
from buttermilk._core.dmrc import get_bm, set_bm

# Initialize with OSB configuration
cfg = init(job="osb_example", overrides=["flows=[osb]"])
set_bm(cfg.bm)
bm = get_bm()
```

### 2. Load OSB Data

```python
from buttermilk._core.config import DataSourceConfig
from buttermilk.data.loaders import create_data_loader

# Create data loader for OSB JSON file
osb_config = DataSourceConfig(
    type="file",
    path="gs://prosocial-public/osb/03_osb_fulltext_summaries.json"
)
loader = create_data_loader(osb_config)
records = list(loader)
```

### 3. Create Vector Store

```python
from buttermilk.data.vector import ChromaDBEmbeddings, InputDocument

# Initialize vector store
vector_store = ChromaDBEmbeddings(
    collection_name="osb_cases",
    persist_directory="./osb_vectorstore",
    embedding_model="gemini-embedding-001",
    dimensionality=3072
)

# Process documents
for record in records:
    input_doc = InputDocument(
        record_id=record.record_id,
        title=f"OSB Case {record.record_id}",
        full_text=record.content,
        metadata=record.metadata or {}
    )
    await vector_store.process(input_doc)
```

### 4. Search Cases

```python
# Semantic search
results = vector_store.collection.query(
    query_texts=["content moderation appeals"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
```

## Configuration Reference

### OSB Flow Configuration

The OSB flow is defined in `/conf/flows/osb.yaml`:

```yaml
osb:
  _target_: buttermilk.runner.flow.Flow
  source: api development
  steps:
    - name: search
      _target_: LLMAgent
      parameters: 
        template: osb
        formatting: json
      variants:
        model: ${llm}
      data: osbcasessummary

  storage:
    cases:
      type: vector
      uri: gs://prosocial-public/osb/03_osb_fulltext_summaries.json
      db:
        type: chromadb
        embeddings: gs://prosocial-public/osb/04_osb_embeddings_vertex-005.json
        model: text-embedding-005
        store: ".chromadb"
```

### OSB Data Source Configuration

Located in `/conf/data/osb.yaml`:

```yaml
osb:
  type: vector
  uri: gs://prosocial-public/osb/03_osb_fulltext_summaries.json
  db:
    type: chromadb
    embeddings: gs://prosocial-public/osb/04_osb_embeddings_vertex-005.json
    model: text-embedding-005
    store: ".chromadb"
```

### BigQuery Integration

For persistent storage in BigQuery (`/conf/data/osb_bigquery.yaml`):

```yaml
tox_train:
  type: bigquery
  project_id: dmrc-analysis
  dataset_id: toxicity
  table_id: osb_drag_toxic_train
  dataset_name: tox_train
  split_type: train
  randomize: true
  batch_size: 1000
  auto_create: true
  clustering_fields: ["record_id", "dataset_name"]
```

## Data Loading

### JSON Data Structure

The OSB dataset is stored as JSON with the following structure:

```json
{
  "id": "case_001",
  "text": "Full case text and decision...",
  "title": "Case Title",
  "metadata": {
    "date": "2024-01-01",
    "jurisdiction": "UK",
    "case_type": "content_moderation"
  }
}
```

### Data Loading Options

#### Option 1: Direct JSON Loading

```python
from buttermilk.data.loaders import JSONLDataLoader
from buttermilk._core.config import DataSourceConfig

config = DataSourceConfig(
    type="file",
    path="gs://prosocial-public/osb/03_osb_fulltext_summaries.json",
    columns={
        "content": "text",
        "record_id": "id"
    }
)
loader = JSONLDataLoader(config)
```

#### Option 2: Using Pre-computed Embeddings

If embeddings are already available:

```python
import json
from pathlib import Path

# Load pre-computed embeddings
embeddings_path = "gs://prosocial-public/osb/04_osb_embeddings_vertex-005.json"
# Implementation depends on embedding format
```

#### Option 3: Streaming Large Datasets

For large datasets, use streaming:

```python
def stream_osb_data(batch_size=100):
    loader = create_data_loader(osb_config)
    batch = []
    
    for record in loader:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch
```

## Vector Store Setup

### ChromaDB Configuration

```python
from buttermilk.data.vector import ChromaDBEmbeddings

vector_store = ChromaDBEmbeddings(
    # Core settings
    collection_name="osb_legal_cases",
    persist_directory="/path/to/vectorstore",
    
    # Embedding settings
    embedding_model="text-embedding-005",  # or "text-embedding-large-exp-03-07"
    dimensionality=3072,  # Vertex AI standard
    
    # Performance settings
    concurrency=20,  # Embedding concurrency
    upsert_batch_size=50,  # ChromaDB batch size
    embedding_batch_size=5,  # Vertex AI batch size
    
    # Storage settings
    arrow_save_dir="/path/to/chunks"  # Parquet storage for chunks
)
```

### Text Chunking Options

```python
from buttermilk.data.vector import DefaultTextSplitter

# Standard chunking for legal text
splitter = DefaultTextSplitter(
    chunk_size=1500,      # Larger chunks for legal context
    chunk_overlap=300     # Preserve context across chunks
)

# Custom chunking for case structure
class LegalTextSplitter(DefaultTextSplitter):
    def __init__(self):
        super().__init__(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " "]  # Legal document structure
        )
```

### Remote Storage and Caching

For remote ChromaDB storage:

```python
# The vector store automatically handles remote caching
vector_store = ChromaDBEmbeddings(
    persist_directory="gs://your-bucket/osb-vectorstore",
    collection_name="osb_cases"
)

# Cache is automatically initialized
await vector_store.ensure_cache_initialized()
```

## Search and Retrieval

### Basic Semantic Search

```python
def search_osb_cases(query: str, n_results: int = 10):
    results = vector_store.collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    for doc, metadata, distance in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        similarity = 1 - distance
        print(f"Similarity: {similarity:.3f}")
        print(f"Case: {metadata.get('document_title')}")
        print(f"Text: {doc[:200]}...")
        print("-" * 40)
```

### Advanced Filtering

```python
# Filter by metadata
filtered_results = vector_store.collection.query(
    query_texts=["platform liability"],
    n_results=5,
    where={
        "case_type": {"$eq": "content_moderation"},
        "jurisdiction": {"$eq": "UK"}
    },
    include=["documents", "metadatas", "distances"]
)

# Complex filters
complex_filter = {
    "$and": [
        {"date": {"$gte": "2023-01-01"}},
        {"$or": [
            {"case_type": {"$eq": "misinformation"}},
            {"case_type": {"$eq": "hate_speech"}}
        ]}
    ]
}
```

### Similarity Thresholds

```python
def search_with_threshold(query: str, threshold: float = 0.7):
    results = vector_store.collection.query(
        query_texts=[query],
        n_results=50,  # Get more to filter
        include=["documents", "metadatas", "distances"]
    )
    
    # Filter by similarity threshold
    filtered = []
    for doc, metadata, distance in zip(*results.values()):
        similarity = 1 - distance
        if similarity >= threshold:
            filtered.append((doc, metadata, similarity))
    
    return filtered
```

## Integration with Buttermilk Flows

### Using OSB Expert Template

```python
from buttermilk.utils.templating import render_template

# Search for relevant cases
search_results = vector_store.collection.query(
    query_texts=["content moderation appeals process"],
    n_results=3,
    include=["documents", "metadatas"]
)

# Prepare data for template
dataset = []
for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
    dataset.append({
        'title': metadata.get('document_title', 'Unknown Case'),
        'text': doc,
        'record_id': metadata.get('document_id', 'unknown')
    })

# Render OSB expert prompt
prompt_context = {
    'dataset': dataset,
    'formatting': 'Provide structured legal analysis',
    'prompt': 'Analyze the appeals process requirements'
}

rendered_prompt = render_template('osb', prompt_context)
```

### Custom Flow Integration

```python
from buttermilk.agents.llm import LLMAgent
from buttermilk._core.agent import AgentInput

class OSBSearchAgent(LLMAgent):
    def __init__(self, vector_store: ChromaDBEmbeddings, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store
    
    async def _process(self, agent_input: AgentInput):
        # Extract query from input
        query = agent_input.parameters.get('query', agent_input.content)
        
        # Search vector store
        results = self.vector_store.collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        # Prepare context for LLM
        context = self._prepare_context(results)
        
        # Process with LLM
        return await super()._process(agent_input)
```

### Batch Processing

```python
async def process_osb_batch(records: list, vector_store: ChromaDBEmbeddings):
    """Process a batch of OSB records through the pipeline"""
    
    # Convert to InputDocuments
    input_docs = [
        InputDocument(
            record_id=record.record_id,
            title=f"OSB Case {record.record_id}",
            full_text=record.content,
            metadata=record.metadata or {}
        )
        for record in records
    ]
    
    # Process through pipeline
    from buttermilk.data.vector import list_to_async_iterator, DocProcessor
    
    doc_iterator = list_to_async_iterator(input_docs)
    processor = DocProcessor(
        _doc_iterator=doc_iterator,
        _processor=vector_store.process,
        concurrency=vector_store.concurrency
    )
    
    successful = 0
    async for processed_doc in processor():
        if processed_doc:
            successful += 1
    
    return successful
```

## Production Deployment

### Performance Optimization

#### Embedding Optimization

```python
# Use larger batch sizes for production
vector_store = ChromaDBEmbeddings(
    embedding_batch_size=25,    # Max for Vertex AI
    concurrency=50,             # Higher concurrency
    upsert_batch_size=100       # Larger ChromaDB batches
)

# Consider using pre-computed embeddings
# to reduce API costs and latency
```

#### Memory Management

```python
# Process in chunks to manage memory
async def process_large_dataset(records, chunk_size=1000):
    total_processed = 0
    
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        processed = await process_osb_batch(chunk, vector_store)
        total_processed += processed
        
        # Optional: garbage collection
        import gc
        gc.collect()
        
        print(f"Processed {total_processed}/{len(records)} records")
    
    return total_processed
```

### Storage Considerations

#### Persistent Storage

```python
# Production configuration
production_config = {
    "collection_name": "osb_production_v1",
    "persist_directory": "/mnt/vectorstore/osb",  # Persistent volume
    "embedding_model": "text-embedding-005",
    "arrow_save_dir": "/mnt/chunks/osb"
}
```

#### Backup Strategy

```python
import shutil
from datetime import datetime

def backup_vectorstore(source_dir: str, backup_base: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{backup_base}/osb_backup_{timestamp}"
    
    shutil.copytree(source_dir, backup_dir)
    print(f"Backup created: {backup_dir}")
    
    return backup_dir

# Schedule regular backups
backup_vectorstore("/mnt/vectorstore/osb", "/mnt/backups")
```

#### Cloud Storage Integration

```python
# Sync to cloud storage
async def sync_to_cloud(local_dir: str, cloud_uri: str):
    from buttermilk.utils.utils import upload_directory
    
    await upload_directory(local_dir, cloud_uri)
    print(f"Synced {local_dir} to {cloud_uri}")

# Example: sync to GCS
await sync_to_cloud(
    "/mnt/vectorstore/osb",
    "gs://your-bucket/vectorstore-backups/"
)
```

### Monitoring and Metrics

```python
import time
from buttermilk._core.log import logger

class OSBVectorStoreMonitor:
    def __init__(self, vector_store: ChromaDBEmbeddings):
        self.vector_store = vector_store
        self.metrics = {
            "searches_performed": 0,
            "documents_indexed": 0,
            "average_search_time": 0.0,
            "embedding_costs": 0.0
        }
    
    def time_search(self, query: str):
        start_time = time.time()
        
        results = self.vector_store.collection.query(
            query_texts=[query],
            n_results=10
        )
        
        search_time = time.time() - start_time
        self.metrics["searches_performed"] += 1
        
        # Update running average
        current_avg = self.metrics["average_search_time"]
        count = self.metrics["searches_performed"]
        self.metrics["average_search_time"] = (
            (current_avg * (count - 1) + search_time) / count
        )
        
        logger.info(f"Search completed in {search_time:.3f}s")
        return results
    
    def log_metrics(self):
        logger.info(f"OSB Vector Store Metrics: {self.metrics}")
```

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

```bash
# Ensure Google Cloud credentials are configured
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Or use gcloud auth
gcloud auth application-default login
```

#### 2. Memory Issues

```python
# Reduce batch sizes for limited memory environments
vector_store = ChromaDBEmbeddings(
    concurrency=5,              # Reduce concurrency
    upsert_batch_size=10,       # Smaller batches
    embedding_batch_size=1      # Process one at a time
)
```

#### 3. Embedding API Limits

```python
import asyncio

# Add delays to respect rate limits
async def rate_limited_embedding(vector_store, docs):
    for doc in docs:
        await vector_store.process(doc)
        await asyncio.sleep(0.1)  # 100ms delay
```

#### 4. ChromaDB Corruption

```python
# Reset corrupted collection
def reset_collection(vector_store: ChromaDBEmbeddings):
    client = vector_store._client
    
    try:
        client.delete_collection(vector_store.collection_name)
        print(f"Deleted collection: {vector_store.collection_name}")
    except Exception as e:
        print(f"Collection didn't exist: {e}")
    
    # Recreate collection
    collection = client.get_or_create_collection(
        vector_store.collection_name,
        embedding_function=vector_store._embedding_function
    )
    vector_store._collection = collection
    print(f"Recreated collection: {vector_store.collection_name}")
```

### Performance Debugging

```python
import cProfile
import pstats

def profile_vector_operations():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your vector operations here
    # e.g., vector_store.process(doc)
    
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Validation and Testing

```python
async def validate_vectorstore(vector_store: ChromaDBEmbeddings):
    """Validate vector store integrity"""
    
    # Check collection exists
    try:
        count = vector_store.collection.count()
        print(f"✅ Collection contains {count} documents")
    except Exception as e:
        print(f"❌ Collection error: {e}")
        return False
    
    # Test search functionality
    try:
        results = vector_store.collection.query(
            query_texts=["test query"],
            n_results=1
        )
        print("✅ Search functionality working")
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False
    
    # Validate embeddings
    if count > 0:
        sample = vector_store.collection.get(
            limit=1,
            include=["embeddings"]
        )
        if sample['embeddings'] and sample['embeddings'][0]:
            embedding_dim = len(sample['embeddings'][0])
            expected_dim = vector_store.dimensionality
            
            if embedding_dim == expected_dim:
                print(f"✅ Embeddings have correct dimensionality: {embedding_dim}")
            else:
                print(f"❌ Embedding dimension mismatch: {embedding_dim} != {expected_dim}")
                return False
    
    return True
```

## API Reference

### ChromaDBEmbeddings Class

```python
class ChromaDBEmbeddings(DataSouce):
    """Handles configuration, embedding model interaction, and ChromaDB connection."""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: str = "text-embedding-005",
        dimensionality: int = 3072,
        concurrency: int = 20,
        upsert_batch_size: int = 10,
        embedding_batch_size: int = 1,
        arrow_save_dir: str = ""
    ):
        """Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to store ChromaDB files
            embedding_model: Vertex AI embedding model name
            dimensionality: Embedding vector dimensionality
            concurrency: Number of concurrent embedding operations
            upsert_batch_size: Batch size for ChromaDB upserts
            embedding_batch_size: Batch size for embedding API calls
            arrow_save_dir: Directory to store chunk parquet files
        """
    
    async def process(self, doc: InputDocument) -> InputDocument | None:
        """Process a document: chunk, embed, and store in ChromaDB.
        
        Args:
            doc: InputDocument to process
            
        Returns:
            Processed InputDocument with embeddings, or None if failed
        """
    
    async def ensure_cache_initialized(self) -> None:
        """Ensure ChromaDB cache is initialized for remote storage."""
    
    def check_document_exists(self, document_id: str) -> bool:
        """Check if a document already exists in the collection.
        
        Args:
            document_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
```

### Data Loading Functions

```python
def create_data_loader(config: DataSourceConfig) -> DataLoader:
    """Factory function to create appropriate DataLoader.
    
    Args:
        config: Data source configuration
        
    Returns:
        Configured DataLoader instance
        
    Raises:
        ValueError: If data source type is not supported
    """

class JSONLDataLoader(DataLoader):
    """Loader for JSONL (JSON Lines) files."""
    
    def __iter__(self) -> Iterator[Record]:
        """Load and yield records from JSONL file."""
```

### Template Rendering

```python
def render_template(template_name: str, context: dict) -> str:
    """Render a Jinja2 template with context.
    
    Args:
        template_name: Name of the template (e.g., 'osb')
        context: Template context variables
        
    Returns:
        Rendered template string
    """
```

### Utility Functions

```python
async def ensure_chromadb_cache(persist_directory: str) -> pathlib.Path:
    """Ensure ChromaDB database files are available locally.
    
    Args:
        persist_directory: Remote path to ChromaDB data
        
    Returns:
        Local path to cached ChromaDB directory
    """

def list_to_async_iterator(items: list[T]) -> AsyncIterator[T]:
    """Convert a list into an asynchronous iterator.
    
    Args:
        items: List of items to iterate over
        
    Yields:
        Items from the list asynchronously
    """
```

---

## Conclusion

This guide provides comprehensive coverage of using Buttermilk's vector database infrastructure with the OSB dataset. The combination of robust data loading, flexible vector storage, and powerful search capabilities makes it an ideal platform for legal document analysis and semantic search applications.

For additional support or questions, refer to the Buttermilk documentation or examine the example notebook at `/examples/osb_vector_example.ipynb`.