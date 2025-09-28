"""LLM processor for pipeline integration.

This module provides the LLMProcessor class that integrates LLMCore into
pipeline workflows by implementing the standard Processor protocol.
"""

from typing import Any, AsyncGenerator

from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord


class LLMProcessor:
    """Processor that integrates LLMCore into pipelines.

    This processor wraps LLMCore to work with the pipeline architecture,
    handling the dict-based flow expected by pipelines while providing
    flexible LLM processing capabilities.

    Attributes:
        llm_core: The LLMCore instance for processing
        output_field: How to handle the LLM output in the pipeline
    """

    def __init__(
        self,
        llm_core: LLMCore,
        output_field: str = "record"
    ):
        """Initialize the LLM processor.

        Args:
            llm_core: Configured LLMCore instance
            output_field: How to handle LLM output:
                - "record": Update the existing record with LLM output
                - "llm_output": Add LLM output as new field alongside record
                - Any other string: Use as custom field name for LLM output
        """
        self.llm_core = llm_core
        self.output_field = output_field

    async def process(self, inputs: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Process inputs through LLMCore and yield result.

        Args:
            inputs: Dictionary containing 'record' and other fields

        Yields:
            Dictionary with processed LLM output
        """
        # Process through LLMCore (it accepts dict inputs directly!)
        async for result in self.llm_core.process(inputs=inputs):

            # Decide how to structure the output
            if self.output_field == "record" and "record" in inputs:
                # Update the existing record with LLM output
                original_record = inputs["record"]

                # If LLM output is a BaseRecord, use it directly
                if isinstance(result.content, BaseRecord):
                    updated_record = result.content
                else:
                    # Otherwise, update the original record's content
                    updated_record = original_record.model_copy(update={
                        "content": result.content,
                        "metadata": {
                            **original_record.metadata,
                            "llm_processing": result.metadata
                        }
                    })

                yield {
                    **inputs,  # Preserve other fields
                    "record": updated_record
                }

            else:
                # Add LLM output as a new field
                field_name = self.output_field if self.output_field != "record" else "llm_output"
                yield {
                    **inputs,  # Preserve original inputs
                    field_name: result.content,
                    "llm_metadata": result.metadata
                }


class SimpleLLMProcessor(LLMProcessor):
    """Simplified LLM processor that creates LLMCore internally.

    This is a convenience class that creates an LLMCore instance
    from basic parameters, making it easier to use in pipeline
    configurations.
    """

    def __init__(
        self,
        model: str,
        template: str,
        output_field: str = "record",
        **llm_core_kwargs
    ):
        """Initialize with basic LLM parameters.

        Args:
            model: LLM model name (e.g., "gpt-4", "claude-3")
            template: Template name to use
            output_field: How to handle LLM output
            **llm_core_kwargs: Additional arguments for LLMCore
        """
        llm_core = LLMCore(
            model=model,
            template=template,
            **llm_core_kwargs
        )
        super().__init__(llm_core, output_field)