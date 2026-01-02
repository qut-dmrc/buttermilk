"""Buttermilk processors for pipeline data processing.

This module provides various processors that implement the Processor protocol
for use in data processing pipelines. Each processor takes dict inputs
containing 'record' and other fields, yielding transformed dict outputs.
"""

from .chromadb_uploader import ChromaDBUploader
from .embeddings import EmbeddingGenerator
from .jmespath_transform import JMESPathTransform
from .parallel import ParallelProcessor
from .unified_processors import GroupchatProcessor, ParameterExpansionProcessor
from .variants import VariantProcessor
from .vertex_batch import VertexBatchProcessor

__all__ = [
    "ChromaDBUploader",
    "EmbeddingGenerator",
    "GroupchatProcessor",
    "JMESPathTransform",
    "ParallelProcessor",
    "ParameterExpansionProcessor",
    "VariantProcessor",
    "VertexBatchProcessor",
]
