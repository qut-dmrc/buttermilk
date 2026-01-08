# RFC #311 Migration Guide: Unified Processor Architecture

**Version:** 1.0 | **Date:** 2025-12-21

## Overview

RFC #311 unifies processor architecture with consistent protocols, type-safe configuration, and `ProcessingContext` for state management.

**Benefits**: Unified interfaces, consistent observability, fail-fast semantics, Pydantic validation.

## What Changed

**Old**: Multiple patterns (`ProcessorCore`, `OrchestratorProcessor`, `EmbeddingGenerator`, `ChromaDBUploader`, `LLMCore`) with inconsistent interfaces.

**New**: Two protocols (`Processor`, `BatchProcessor`), unified `ProcessingContext`, base classes (`UnifiedProcessor`, `UnifiedBatchProcessor`), type-safe `ProcessorConfig` with discriminated unions.

## Processor Types

| Type | Purpose | Replaces |
|------|---------|----------|
| `llm` | LLM transformations | Direct `LLMCore` usage |
| `groupchat` | Multi-agent flows | `OrchestratorProcessor` |
| `embedding` | Batch embeddings | `EmbeddingGenerator` |
| `chromadb` | Vector DB uploads | `ChromaDBUploader` |
| `filter` | JMESPath filtering | - |
| `transform` | JMESPath transforms | - |
| `expander` | 1:N record expansion | - |
| `shell` | Shell commands | - |

## Configuration

**Before**:
```yaml
steps:
  - class: buttermilk.processors.EmbeddingGenerator
    config:
      model: gemini-embedding-001
```

**After**:
```yaml
processors:
  - type: embedding
    name: embedder
    embedding_model: gemini-embedding-001
    batch_size: 32
```

**Changes**: Type discriminators, inline config fields, strict validation.

## Migration Steps

### 1. OrchestratorProcessor → GroupchatProcessor

```yaml
# Before
- class: buttermilk.processors.OrchestratorProcessor
  config:
    flow_name: moderation

# After
- type: groupchat
  name: moderation_flow
  flow_name: moderation
  flow_config: !include flows/moderation.yaml
```

### 2. EmbeddingGenerator → EmbeddingProcessor

```yaml
# Before
- class: buttermilk.processors.EmbeddingGenerator
  config:
    model: gemini-embedding-001

# After
- type: embedding
  name: embedder
  embedding_model: gemini-embedding-001
  dimensionality: 3072
```

### 3. ChromaDBUploader → ChromaDBProcessor

```yaml
# Before
- class: buttermilk.processors.ChromaDBUploader
  config:
    collection_name: docs

# After
- type: chromadb
  name: vector_store
  collection_name: docs
  persist_directory: gs://bucket/chromadb
```

### 4. LLMCore → LLMProcessor (pipelines)

```yaml
# New (for pipeline integration)
- type: llm
  name: summarizer
  model: gpt-4
  template: "Summarize: {text}"
  output_col: summary
```

## CLI Changes

Batch mode unified with pipeline mode (same execution path):

```bash
# Both use GroupchatProcessor internally
python -m buttermilk.runner.cli run=batch run.flow=trans
python -m buttermilk.runner.cli run=pipeline run.flow=trans
```

## Backward Compatibility

| Component | Status | Action |
|-----------|--------|--------|
| `OrchestratorProcessor` | **Deprecated** | **Migrate** |
| `EmbeddingGenerator` | Compatible | Optional |
| `ChromaDBUploader` | Compatible | Optional |
| `ProcessorCore` | Compatible | None |
| Batch CLI | Compatible | None |

**Strategy**: Gradual migration. Old and new processors coexist.

## Checklist

**Pre-Migration**:
- [ ] Read guide, identify processors, backup configs

**Migration**:
- [ ] `OrchestratorProcessor` → `GroupchatProcessor`
- [ ] `EmbeddingGenerator` → `EmbeddingProcessor`
- [ ] `ChromaDBUploader` → `ChromaDBProcessor`
- [ ] Update custom processors for `ProcessingContext`

**Validation**:
- [ ] Add type discriminators, validate configs, test pipelines

## Troubleshooting

- **"Processor type not found"**: Import processor module
- **"Missing required field"**: Check Pydantic config
- **Context attribute error**: Use `context.metadata`

## Example Config

```yaml
pipeline:
  name: moderation_pipeline
  processors:
    - type: embedding
      name: embedder
      embedding_model: gemini-embedding-001
      batch_size: 32
    - type: groupchat
      name: moderation
      flow_name: content_moderation
      flow_config: !include flows/moderation.yaml
    - type: chromadb
      name: vector_store
      collection_name: documents
      persist_directory: gs://bucket/chromadb
```

## Resources

- Processor code: `buttermilk/processors/unified_processors.py`
- Config models: `buttermilk/_core/processor_config.py`
- Tests: `tests/test_unified_processor.py`
