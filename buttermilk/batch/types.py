"""Common types for batch processing."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class BatchRequest(BaseModel):
    """A single request in a batch job.

    Attributes:
        custom_id: Unique identifier for mapping results back to input
        record_id: Original record ID from the source data
        messages: The messages to send to the model (LiteLLM format)
        model: Model identifier used for this request (for result analysis)
        variant: Variant identifier (e.g., label string or structured dict)
        processor_index: Index of this processor in a multi-processor pipeline
        response_schema: Pre-resolved JSON schema dict for structured output.
            When set, converters will include provider-specific structured output
            parameters in the batch request (e.g., generationConfig for Gemini,
            tool-based extraction for Claude).
    """

    custom_id: str
    record_id: str
    messages: list[dict[str, Any]]
    model: str | None = None
    variant: str | dict[str, Any] | None = None
    processor_index: int | None = None
    response_schema: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
    )


class BatchResult(BaseModel):
    """Result from a batch job request.

    Attributes:
        custom_id: The custom_id from the request
        record_id: Original record ID
        response: The model's response content
        error: Error message if request failed
        usage: Token usage information
        model: Model identifier used for this request (from BatchRequest)
        variant: Variant identifier (from BatchRequest)
        processor_index: Processor index (from BatchRequest)
    """

    custom_id: str
    record_id: str
    response: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    variant: str | dict[str, Any] | None = None
    processor_index: int | None = None
    cost_usd: float | None = None

    @property
    def composite_key(self) -> str:
        """Generate a composite key for unique identification across variants.

        Format: {record_id}[_{variant}][_{model}][_{processor_index}]
        Only includes non-None components.

        Returns:
            str: Composite key for unique identification
        """
        parts = [self.record_id]
        if self.variant:
            if isinstance(self.variant, dict):
                # For dict variants, generate a stable string representation
                # Focus on identifying keys/values
                variant_str = json.dumps(self.variant, sort_keys=True)
                parts.append(variant_str)
            else:
                parts.append(str(self.variant))
        if self.model:
            parts.append(self.model)
        if self.processor_index is not None:
            parts.append(str(self.processor_index))
        return "_".join(parts)
