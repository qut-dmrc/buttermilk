"""Buttermilk processors for pipeline data processing.

This module provides various processors that implement the Processor protocol
for use in data processing pipelines. Each processor takes dict inputs
containing 'record' and other fields, yielding transformed dict outputs.
"""

from .chromadb_uploader import ChromaDBUploader
from .embeddings import EmbeddingGenerator
from .jmespath_transform import JMESPathTransform
from .orchestrator_processor import OrchestratorProcessor  # Deprecated, kept for backward compatibility
from .parallel import ParallelProcessor
from .unified_processors import GroupchatProcessor
from .variants import VariantProcessor

__all__ = [
    "ChromaDBUploader",
    "EmbeddingGenerator",
    "GroupchatProcessor",  # Preferred for multi-agent flows
    "JMESPathTransform",
    "OrchestratorProcessor",  # Deprecated: Use GroupchatProcessor instead
    "ParallelProcessor",
    "VariantProcessor",
]
