"""Buttermilk processors for pipeline data processing.

This module provides various processors that implement the Processor protocol
for use in data processing pipelines. Each processor takes dict inputs
containing 'record' and other fields, yielding transformed dict outputs.
"""

from .batch_accumulator import BatchAccumulator
from .chromadb_uploader import ChromaDBUploader
from .embeddings import EmbeddingGenerator
from .jmespath_transform import JMESPathTransform
from .parallel import ParallelProcessor
from .unified_processors import GroupchatProcessor, LLMProcessor, ParameterExpansionProcessor
from .variants import VariantProcessor
from .vertex_batch import BatchLLMProcessor, OpenAIBatchProcessor, VertexBatchProcessor

__all__ = [
    "BatchAccumulator",
    "BatchLLMProcessor",
    "ChromaDBUploader",
    "EmbeddingGenerator",
    "GroupchatProcessor",
    "JMESPathTransform",
    "LLMProcessor",
    "OpenAIBatchProcessor",
    "ParallelProcessor",
    "ParameterExpansionProcessor",
    "VariantProcessor",
    "VertexBatchProcessor",
]
