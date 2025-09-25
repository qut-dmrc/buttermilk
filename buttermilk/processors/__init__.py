"""Buttermilk processors for pipeline data processing.

This module provides various processors that implement the Processor protocol
for use in data processing pipelines. Each processor takes BaseRecord objects
as input and yields transformed BaseRecord objects as output.
"""

from buttermilk.processors.llm import LLMProcessor

__all__ = ["LLMProcessor"]