"""LLM Processor for Buttermilk pipelines.

This module provides the LLMProcessor class that enables LLM operations
within data processing pipelines. It implements the Processor protocol
to work seamlessly with the PipelineOrchestrator.

Unlike LLMAgent which handles autogen sessions and agent-specific messaging,
LLMProcessor is designed for simple, stateless record transformations using
LLMs. It processes BaseRecord objects through templates and LLM calls,
enriching them with generated content.
"""

import time
from typing import Any, AsyncGenerator, Optional

import pydantic
from autogen_core.tools import Tool

from buttermilk import logger
from buttermilk._core.contract import ErrorEvent, ExecutionTrace
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

        # Store additional config
        self.config = kwargs

        # Initialize trace writer (lazy loading)
        self._trace_writer = None

        logger.debug(
            f"LLMProcessor initialized: model={parameters.get('model')}, "
            f"template={parameters.get('template')}, "
            f"input_field={input_field}, output_field={output_field}"
        )

    @property
    def trace_writer(self):
        """Lazy load trace writer."""
        if self._trace_writer is None:
            from buttermilk.utils.trace_writer import get_trace_writer
            self._trace_writer = get_trace_writer()
        return self._trace_writer

    async def process(self, record: BaseRecord) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord through the LLM and yield the enriched result.

        This method implements the Processor protocol for pipeline compatibility.
        It extracts the specified input field, processes it through the LLM,
        and updates the record with the result.

        Args:
            record: The BaseRecord to process

        Yields:
            The same record, enriched with LLM-generated content in the output field.
            If processing fails, yields the record with error information.
        """
        start_time = time.time()
        trace = None

        try:
            # Extract input data from the record
            if hasattr(record, self.input_field):
                input_data = getattr(record, self.input_field)
            elif hasattr(record, "data") and isinstance(record.data, dict):
                # Try to get from data dict if it exists
                input_data = record.data.get(self.input_field)
            else:
                # Try direct dictionary access if record supports it
                input_data = record.get(self.input_field) if hasattr(record, "get") else None

            if input_data is None:
                raise ValueError(f"Field '{self.input_field}' not found in record {record.record_id}")

            # Prepare inputs for LLM - support both string and dict inputs
            if isinstance(input_data, dict):
                # Make a copy to avoid modifying the original dict
                llm_inputs = input_data.copy()
            else:
                # Wrap non-dict data in a standard format
                llm_inputs = {self.input_field: input_data}

            # Extract any existing trace ID for correlation
            parent_trace_id = record.metadata.get("trace_id") if hasattr(record, "metadata") else None

            logger.debug(f"Processing record {record.record_id} through LLM")

            # Process through LLM
            llm_result = await self.llm_core.process_with_llm(
                inputs=llm_inputs,
                context=None,  # Pipelines typically don't maintain conversation context
                records=[record],  # Pass the record itself for template access
                parent_trace_id=parent_trace_id
            )

            # Update record with the result
            if hasattr(record, self.output_field):
                # Direct attribute
                setattr(record, self.output_field, llm_result.content)
            elif hasattr(record, "data") and isinstance(record.data, dict):
                # Store in data dict
                record.data[self.output_field] = llm_result.content
            elif hasattr(record, "__setitem__"):
                # Dictionary-like interface
                record[self.output_field] = llm_result.content
            else:
                # Fall back to creating a new attribute
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

            # Create ExecutionTrace for observability
            duration_ms = (time.time() - start_time) * 1000
            trace = ExecutionTrace(
                call_id=llm_result.trace_id,
                agent_info={
                    "component_name": "LLMProcessor",
                    "execution_type": "processor",
                    "config": self.parameters,
                },
                inputs=llm_inputs,
                outputs=llm_result.content,
                messages=llm_result.messages,
                parameters=self.parameters,
                record={
                    "record_id": record.record_id,
                    "dataset_name": getattr(record, 'dataset_name', None),
                    "split_type": getattr(record, 'split_type', None),
                },
                metadata={
                    **llm_result.metadata,
                    **llm_result.template_metadata,
                    "duration_ms": duration_ms,
                    "input_field": self.input_field,
                    "output_field": self.output_field,
                },
                parent_call_id=parent_trace_id,
            )

            # Emit trace if trace writer is available
            if hasattr(self, 'trace_writer') and self.trace_writer:
                try:
                    await self.trace_writer.add(trace)
                except Exception as e:
                    logger.warning(f"Failed to emit trace: {e}")

            yield record

        except ProcessingError as e:
            # Handle processing errors gracefully
            logger.error(f"LLM processing failed for record {record.record_id}: {e}")

            # Add error to record
            error_event = ErrorEvent(
                content=f"LLM processing failed: {e}",
                source="LLMProcessor"
            )

            if hasattr(record, "error"):
                if not isinstance(record.error, list):
                    record.error = []
                record.error.append(error_event)
            else:
                record.error = [error_event]

            # Store error in metadata
            if hasattr(record, "metadata"):
                if not isinstance(record.metadata, dict):
                    record.metadata = {}
                record.metadata["llm_error"] = str(e)

            # Create error trace
            duration_ms = (time.time() - start_time) * 1000
            error_trace = ExecutionTrace(
                agent_info={
                    "component_name": "LLMProcessor",
                    "execution_type": "processor",
                    "config": self.parameters,
                },
                inputs=llm_inputs if 'llm_inputs' in locals() else None,
                record={
                    "record_id": record.record_id,
                    "dataset_name": getattr(record, 'dataset_name', None),
                    "split_type": getattr(record, 'split_type', None),
                },
                error={
                    "event": str(e),
                    "details": {"error_type": type(e).__name__}
                },
                metadata={
                    "duration_ms": duration_ms,
                    "input_field": self.input_field,
                    "output_field": self.output_field,
                },
            )

            # Emit error trace if trace writer is available
            if hasattr(self, 'trace_writer') and self.trace_writer:
                try:
                    await self.trace_writer.add(error_trace)
                except Exception as te:
                    logger.warning(f"Failed to emit error trace: {te}")

            # Yield the record even on error (with error information)
            yield record

        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error processing record {record.record_id}: {e}", exc_info=True)

            # Add error to record
            error_event = ErrorEvent(
                content=f"Unexpected error: {e}",
                source="LLMProcessor"
            )

            if hasattr(record, "error"):
                if not isinstance(record.error, list):
                    record.error = []
                record.error.append(error_event)
            else:
                record.error = [error_event]

            # Store error in metadata
            if hasattr(record, "metadata"):
                if not isinstance(record.metadata, dict):
                    record.metadata = {}
                record.metadata["llm_error"] = f"Unexpected error: {e}"

            # Yield the record with error information
            yield record