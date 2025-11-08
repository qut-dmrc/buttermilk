# OSB Vector Database Guide

The Oversight Board (OSB) dataset contains case summaries and decisions related to online safety, content moderation, and platform responsibility. This guide provides comprehensive documentation for working with the Oversight Board dataset using Buttermilk's vector database infrastructure.

## Quick Start

### Load OSB Data

```python
from buttermilk._core.config import DataSourceConfig
from buttermilk.data.loaders import create_data_loader

# Create data loader for OSB JSON file
osb_config = DataSourceConfig(
    type="file", path="gs://prosocial-public/osb/03_osb_fulltext_summaries.json"
)
loader = create_data_loader(osb_config)
records = list(loader)
```

### Create Vector Store

```python
from buttermilk.data.vector import ChromaDBEmbeddings, InputDocument

# Initialize vector store
vector_store = ChromaDBEmbeddings(
    collection_name="osb_cases",
    persist_directory="./osb_vectorstore",
    embedding_model="gemini-embedding-001",
    dimensionality=3072,
)

# Process documents
for record in records:
    input_doc = InputDocument(
        record_id=record.record_id,
        title=f"OSB Case {record.record_id}",
        full_text=record.content,
        metadata=record.metadata or {},
    )
    await vector_store.process(input_doc)
```

### Search Cases

```python
# Semantic search
results = vector_store.collection.query(
    query_texts=["content moderation appeals"],
    n_results=5,
    include=["documents", "metadatas", "distances"],
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
