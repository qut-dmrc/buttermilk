"""LLM Processor for Buttermilk pipelines.

This module provides the LLMProcessor class that enables LLM operations
within data processing pipelines. It implements the Processor protocol
to work seamlessly with the PipelineOrchestrator.

Unlike LLMAgent which handles autogen sessions and agent-specific messaging,
LLMProcessor is designed for simple, stateless record transformations using
LLMs. It processes BaseRecord objects through templates and LLM calls,
enriching them with generated content.
"""

from typing import Any, AsyncGenerator, Optional

import pydantic
from autogen_core.tools import Tool

from buttermilk import logger
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord


class LLMProcessor:
    """Pipeline processor that uses LLM for record transformation.

    This processor takes BaseRecord objects, extracts specified fields,
    processes them through an LLM using templates, and updates the records
    with the results. It's designed for batch processing scenarios where
    you need to enrich records with LLM-generated content.

    Configuration:
        parameters: Dict containing LLM configuration:
            - model: Name of the LLM model to use (required)
            - template: Name of the template to render (required)
            - temperature: Optional temperature setting
            - fail_on_unfilled_parameters: Whether to fail on missing template vars
        input_field: Field name to extract from record for processing (default: "content")
        output_field: Field name to store results in (default: "processed")
        output_model: Optional Pydantic model for structured output
        tools: Optional list of tools the LLM can use

    Example:
        ```python
        processor = LLMProcessor(
            parameters={
                "model": "gemini25flash",
                "template": "summarize_text"
            },
            input_field="full_text",
            output_field="summary"
        )

        async for enriched_record in processor.process(record):
            print(enriched_record.summary)
        ```
    """

    def __init__(
        self,
        parameters: dict[str, Any],
        input_field: str = "content",
        output_field: str = "processed",
        output_model: Optional[type[pydantic.BaseModel]] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs
    ):
        """Initialize the LLM processor with configuration.

        Args:
            parameters: LLM configuration including model and template
            input_field: Name of the field to extract from records for processing
            output_field: Name of the field to store results in
            output_model: Optional Pydantic model for structured output
            tools: Optional list of tools for the LLM
            **kwargs: Additional configuration (for future extensions)
        """
        self.parameters = parameters
        self.input_field = input_field
        self.output_field = output_field
        self.output_model = output_model

        # Initialize the shared LLM core
        self.llm_core = LLMCore(
            parameters=parameters,
            output_model=output_model,
            tools=tools or []
        )

        logger.debug(
            f"LLMProcessor initialized: model={parameters.get('model')}, "
            f"template={parameters.get('template')}, "
            f"input_field={input_field}, output_field={output_field}"
        )

    async def process(self, record: BaseRecord) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord through LLMCore and yield the enriched result.

        This is now a thin wrapper that extracts input data, calls LLMCore.process(),
        and updates the record with the result. LLMCore handles all the complex
        template processing, LLM calling, and tracing.

        Args:
            record: The BaseRecord to process

        Yields:
            The same record, enriched with LLM-generated content in the output field.

        Raises:
            ProcessingError: If processing fails (fail-fast semantics)
        """
        try:
            # Extract input data from the record
            if hasattr(record, self.input_field):
                input_data = getattr(record, self.input_field)
            elif hasattr(record, "data") and isinstance(record.data, dict):
                input_data = record.data.get(self.input_field)
            elif hasattr(record, "get"):
                input_data = record.get(self.input_field)
            else:
                input_data = None

            if input_data is None:
                raise ProcessingError(f"Field '{self.input_field}' not found in record {record.record_id}")

            # Prepare inputs for LLMCore
            if isinstance(input_data, dict):
                llm_inputs = input_data.copy()
            else:
                llm_inputs = {self.input_field: input_data}

            # Add the record itself for template access
            llm_inputs['record'] = record

            # Extract parent trace ID for correlation
            parent_trace_id = record.metadata.get("trace_id") if hasattr(record, "metadata") else None

            logger.debug(f"Processing record {record.record_id} through LLMCore")

            # Process through LLMCore (handles tracing internally)
            async for llm_result in self.llm_core.process(
                inputs=llm_inputs,
                parent_trace_id=parent_trace_id,
                component_name="LLMProcessor"
            ):
                # Update record with the result
                if hasattr(record, self.output_field):
                    setattr(record, self.output_field, llm_result.content)
                elif hasattr(record, "data") and isinstance(record.data, dict):
                    record.data[self.output_field] = llm_result.content
                elif hasattr(record, "__setitem__"):
                    record[self.output_field] = llm_result.content
                else:
                    setattr(record, self.output_field, llm_result.content)

                # Store LLM metadata in record metadata
                if hasattr(record, "metadata"):
                    if not isinstance(record.metadata, dict):
                        record.metadata = {}
                    record.metadata[f"llm_{self.output_field}"] = {
                        "trace_id": llm_result.trace_id,
                        **llm_result.metadata,
                        **llm_result.template_metadata
                    }

                logger.debug(
                    f"Record {record.record_id} processed successfully. "
                    f"Output stored in '{self.output_field}'"
                )

                yield record
                return  # Only process once

        except ProcessingError:
            # Let ProcessingErrors propagate (fail-fast)
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise ProcessingError(f"Unexpected error processing record {record.record_id}: {e}") from e