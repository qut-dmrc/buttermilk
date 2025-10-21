"""Buttermilk processors for pipeline data processing.

This module provides various processors that implement the Processor protocol
for use in data processing pipelines. Each processor takes dict inputs
containing 'record' and other fields, yielding transformed dict outputs.
"""

from .embeddings import EmbeddingGenerator
from .chromadb_uploader import ChromaDBUploader
from .jmespath_transform import JMESPathTransform

__all__ = [
    "EmbeddingGenerator",
    "ChromaDBUploader",
    "JMESPathTransform",
]
