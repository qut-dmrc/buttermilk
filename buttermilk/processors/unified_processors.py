"""Standard Unified Processors implementations.

This module contains concrete implementations of standard processors:
- LLMProcessor: Runs LLM-based transformations on records.
- GroupchatProcessor: Runs a group chat session.
- ExpanderProcessor: Expands a record into multiple records (1:N).
- ParameterExpansionProcessor: Expands a record based on cartesian product of variants.
- TransformProcessor: Applies JMESPath transformations to records.
- ShellProcessor: Executes shell commands with placeholder substitution.
- FilterProcessor: Filters records based on JMESPath criteria.
- EmbeddingProcessor: Generates embeddings for chunks in batch.
- ChromaDBProcessor: Uploads records with embeddings to ChromaDB.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import chromadb
import jmespath
from google import genai
from jmespath.exceptions import JMESPathError
from pydantic import Field, PrivateAttr

# For ChromaDBProcessor remote storage support
# Import bm for session_info access (same pattern as chromadb_uploader.py)
from buttermilk import bm, logger
from buttermilk._core.contract import ExecutionTrace, TaskProcessingComplete
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llm_core import LLMCore
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord, RunRequest
from buttermilk._core.unified_batch_processor import UnifiedBatchProcessor
from buttermilk._core.unified_processor import UnifiedProcessor
from buttermilk.data.vector import _sanitize_metadata_for_chroma
from buttermilk.runner.flowrunner import OrchestratorFactory
from buttermilk.utils.utils import scrub_serializable, upload_chromadb_cache


class LLMProcessor(UnifiedProcessor):
    """Unified processor for LLM-based transformations.

    Uses LLMCore internally to handle template rendering and LLM calls.
    Processes single records through LLM inference and enriches them with outputs.

    This processor enables LLM-based transformations within the unified processor
    architecture while maintaining proper observability and fail-fast semantics.
    """

    model: str = Field(..., description="LLM model identifier")
    template: str = Field(..., description="Jinja2 template for prompts")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    output_col: str = Field(default="llm_output", description="Output column name")
    input_variables: dict[str, Any] = Field(default_factory=dict, description="Static variables for the prompt")
    output_model: str | None = Field(default=None, description="Pydantic model path for structured output")
    fail_on_unfilled_parameters: bool = Field(default=True, description="Fail if template parameters are unfilled")

    _llm_core: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Create LLMCore instance after Pydantic initialization."""
        self._llm_core = LLMCore(
            model=self.model,
            template=self.template,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            output_col=self.output_col,
            template_vars=self.input_variables,
            output_model=self.output_model,
            fail_on_unfilled_parameters=self.fail_on_unfilled_parameters,
        )

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a record through LLM inference.

        Args:
            context: Processing context containing the record to process

        Yields:
            The record enriched with LLM output in the configured output column

        Raises:
            ProcessingError: If LLM processing fails
        """
        record_id = context.record.record_id

        logger.debug(
            "LLMProcessor starting",
            record_id=record_id,
            model=self.model,
            template=self.template,
        )

        # Build template variables by flattening record metadata
        # This allows templates to access both record fields and metadata fields directly
        record_dict = context.record.model_dump()
        template_vars = {
            **record_dict.get("metadata", {}),  # Flatten metadata fields
            **{k: v for k, v in record_dict.items() if k != "metadata"},  # Top-level fields
        }

        # Use LLMCore.process_with_llm() for LLM inference
        # Trace emission is handled by UnifiedProcessor base class
        llm_result = await self._llm_core.process_with_llm(
            template_vars=template_vars,
            record=context.record,
            parent_trace_id=context.session_id,
        )

        # Check for errors
        if llm_result.error:
            from buttermilk._core.exceptions import ProcessingError

            raise ProcessingError(f"LLM processing failed: {llm_result.error}")

        # Enrich record with LLM output
        enriched_metadata = {
            **(context.record.metadata if context.record.metadata else {}),
            f"llm_{self.name or 'processor'}": llm_result.metadata,
        }

        enriched_record = context.record.model_copy(
            update={
                self.output_col: llm_result.content,
                "metadata": enriched_metadata,
            }
        )

        logger.debug(
            "LLMProcessor completed",
            record_id=record_id,
            output_col=self.output_col,
        )

        yield enriched_record


class GroupchatProcessor(UnifiedProcessor):
    """Wraps an Orchestrator to run multi-agent conversations as a unified processor.

    Creates a fresh orchestrator instance per record to ensure state isolation.
    Collects ExecutionTrace outputs via callback and enriches record metadata with results.

    This processor enables complex multi-agent flows to be used within the unified
    processor architecture while maintaining proper isolation and observability.
    """

    flow_name: str = Field(..., description="Name of the flow to execute")
    flow_config: Any = Field(..., description="Flow configuration")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameters for orchestrator")
    collect_traces: bool = Field(default=True, description="Collect ExecutionTrace outputs")

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Execute orchestrator on a record and enrich with results.

        Creates a fresh orchestrator instance for state isolation, builds RunRequest
        from record, collects traces via callback, and enriches record metadata.

        Args:
            context: Processing context containing the record to process

        Yields:
            The input record enriched with orchestrator outputs in metadata

        Raises:
            Exception: Re-raises any orchestrator errors for pipeline error handling
        """
        record_id = context.record.record_id

        logger.debug(
            "GroupchatProcessor starting",
            record_id=record_id,
            flow_name=self.flow_name,
        )

        # Create fresh orchestrator for each record (state isolation)
        orchestrator = OrchestratorFactory.create_orchestrator(self.flow_config, self.flow_name)

        # Collect ExecutionTrace outputs and TaskProcessingComplete errors via callback
        traces: list[ExecutionTrace] = []
        task_errors: list[TaskProcessingComplete] = []

        async def collect_callback(message: Any) -> None:
            """Callback to collect ExecutionTrace and error signals from orchestrator."""
            if isinstance(message, ExecutionTrace):
                traces.append(message)
            elif isinstance(message, TaskProcessingComplete) and message.is_error:
                task_errors.append(message)

        # Use ui_callback from context if available, otherwise use collect_callback
        callback = context.ui_callback if context.ui_callback else collect_callback

        # If context has ui_callback but we need to collect traces, chain callbacks
        if context.ui_callback and self.collect_traces:

            async def chained_callback(message: Any) -> None:
                """Chain both callbacks."""
                await context.ui_callback(message)
                await collect_callback(message)

            callback = chained_callback
        elif self.collect_traces:
            callback = collect_callback
        else:
            callback = context.ui_callback

        # Create RunRequest from record with metadata merged into inputs
        base_inputs = {
            "record_id": record_id,
            "record": context.record.model_dump() if hasattr(context.record, "model_dump") else context.record,
        }
        merged_inputs = self._merge_metadata_to_inputs(base_inputs, context.record.metadata)

        run_request = RunRequest(
            flow=self.flow_name,
            inputs=merged_inputs,
            parameters=self.parameters,
            callback_to_ui=callback,
        )

        try:
            # Run orchestrator (returns None, results flow through callback)
            await orchestrator.run(request=run_request)

            # Check for errors from TaskProcessingComplete signals (reliable source)
            # These are always published, even when exceptions prevent ExecutionTrace creation
            if task_errors:
                error_details = [f"{t.agent_id}: {t.error or 'unknown error'}" for t in task_errors]
                raise ProcessingError(f"Orchestrator had {len(task_errors)} agent error(s): {'; '.join(error_details)}")

            # Build outputs from collected traces
            outputs = []
            for trace in traces:
                if trace.outputs is not None:
                    outputs.append(trace.outputs)

            # Check if we got any meaningful outputs - no outputs likely means early termination
            if not outputs:
                raise ProcessingError("Orchestrator completed but produced no outputs (early termination or all agents failed)")

            # Enrich record metadata with orchestrator results
            enriched_metadata = {
                **(context.record.metadata if context.record.metadata else {}),
                "groupchat": {
                    "status": "processed",
                    "flow_name": self.flow_name,
                    "trace_count": len(traces),
                    "outputs": outputs,
                },
            }

            logger.debug(
                "GroupchatProcessor completed",
                record_id=record_id,
                flow_name=self.flow_name,
                trace_count=len(traces),
                output_count=len(outputs),
            )

            yield context.record.model_copy(update={"metadata": enriched_metadata})

        except Exception as e:
            logger.error(
                "GroupchatProcessor failed",
                record_id=record_id,
                flow_name=self.flow_name,
                error=str(e),
            )
            # Re-raise to let pipeline handle the error
            raise

    def _merge_metadata_to_inputs(
        self,
        base_inputs: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge record metadata into inputs dict for agent access.

        Merges serializable metadata values into inputs, allowing
        ParameterExpansionProcessor to pass parameters to downstream agents.

        Args:
            base_inputs: Base inputs dict with record_id and record
            metadata: Record metadata to merge (or None)

        Returns:
            Merged inputs dict with metadata values added
        """
        if not metadata:
            return base_inputs

        result = dict(base_inputs)
        reserved_keys = {"record_id", "record"}

        for key, value in metadata.items():
            # Skip reserved keys (base_inputs takes precedence)
            if key in reserved_keys:
                continue

            # Skip non-serializable values (classes, functions, etc.)
            if callable(value) or isinstance(value, type):
                continue

            result[key] = value

        return result


class ExpanderProcessor(UnifiedProcessor):
    """Expands a single record into multiple records based on a list field."""

    field_to_expand: str = Field(..., description="Name of the list field to expand into multiple records")

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        field_name = self.field_to_expand

        # Access the field from the record (attribute or metadata)
        values = getattr(context.record, field_name, None)
        if values is None:
            values = context.record.metadata.get(field_name)

        if not isinstance(values, list):
            logger.warning(
                f"Field '{field_name}' is not a list, cannot expand. Skipping expansion.", record_id=context.record.record_id, value_type=type(values)
            )
            yield context.record
            return

        # Expand logic
        for i, value in enumerate(values):
            # Create updates dictionary
            updates = {"record_id": f"{context.record.record_id}_{i}"}

            # Prepare metadata update
            # We copy existing metadata to avoid mutating the original record's metadata if it's shared
            new_metadata = context.record.metadata.copy()
            new_metadata["expansion_source_id"] = context.record.record_id
            new_metadata["expansion_index"] = i

            if hasattr(context.record, field_name):
                updates[field_name] = value
            else:
                new_metadata[field_name] = value

            updates["metadata"] = new_metadata

            # Create new record with updates
            expanded_record = context.record.model_copy(update=updates)

            yield expanded_record


class ParameterExpansionProcessor(UnifiedProcessor):
    """Expands record into N records based on cartesian product of variants.

    Takes a record and expands it into multiple records by computing the
    cartesian product of all parameter variants. Each variant combination
    becomes a separate record.
    """

    variants: dict[str, list[Any] | Any] = Field(..., description="Parameter variants to expand into cartesian product")

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Expand record into multiple records based on variants.

        Args:
            context: Processing context containing the record to expand

        Yields:
            BaseRecord: One record per variant combination
        """
        from buttermilk.utils.utils import expand_dict

        record = context.record

        # Generate cartesian product of variants
        variant_combinations = expand_dict(self.variants)

        if not variant_combinations or variant_combinations == [{}]:
            yield record
            return

        for combo in variant_combinations:
            # Build variant suffix for tracking (but NOT for record_id - record_id is immutable)
            suffix_parts = [f"{k}={v}" for k, v in sorted(combo.items())]
            variant_suffix = "_".join(suffix_parts)

            # Store variant info in metadata only - record_id stays unchanged
            new_metadata = {
                **(record.metadata if record.metadata else {}),
                **combo,  # Variant parameters as flat keys
                "variant_suffix": variant_suffix,  # For deduplication/tracking if needed
            }

            expanded_record = record.model_copy(update={"metadata": new_metadata})
            yield expanded_record


class TransformProcessor(UnifiedProcessor):
    """Applies JMESPath expressions to transform records.

    Evaluates a JMESPath expression against the record and stores the result
    in context.metadata. This enables declarative data extraction and transformation
    without custom Python code.

    Features:
    - Compiles JMESPath expression once during initialization for performance
    - Stores transformation result in context metadata
    - Fail-fast on invalid JMESPath expressions
    - Graceful handling when expression returns None
    """

    expression: str = Field(..., description="JMESPath expression for transformation")
    output_field: str = Field(default="transformed", description="Output field name in context metadata")

    _compiled_expression: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Compile JMESPath expression after Pydantic initialization."""
        try:
            self._compiled_expression = jmespath.compile(self.expression)
        except JMESPathError as e:
            logger.error(
                "Invalid JMESPath expression",
                expression=self.expression,
                error=str(e),
            )
            raise ValueError(f"Invalid JMESPath expression: {self.expression}") from e

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Apply JMESPath transformation and store result in context metadata.

        Args:
            context: Processing context containing the record to transform

        Yields:
            The original record (transformation result is stored in context.metadata)

        Raises:
            ValueError: If JMESPath expression evaluation fails
        """
        logger.debug(
            "Applying JMESPath transformation",
            record_id=context.record.record_id,
            expression=self.expression,
            output_field=self.output_field,
        )

        # Convert record to dict for JMESPath processing
        record_dict = context.record.model_dump()

        try:
            # Apply JMESPath expression
            result = self._compiled_expression.search(record_dict)

            if result is not None:
                # Store result in context metadata
                context.update_metadata(self.output_field, result)

                logger.debug(
                    "JMESPath transformation complete",
                    record_id=context.record.record_id,
                    output_field=self.output_field,
                    result_type=type(result).__name__,
                )
            else:
                logger.debug(
                    "JMESPath expression returned None, no metadata stored",
                    record_id=context.record.record_id,
                    expression=self.expression,
                )

        except Exception as e:
            logger.error(
                "Error applying JMESPath expression",
                record_id=context.record.record_id,
                expression=self.expression,
                error=str(e),
            )
            raise ValueError(f"Error applying JMESPath expression '{self.expression}': {str(e)}") from e

        # Yield the original record (metadata is stored in context)
        yield context.record


class ShellProcessor(UnifiedProcessor):
    """Executes shell commands with placeholder substitution.

    Features:
    - Executes shell commands with timeout protection
    - Replaces {record_id} placeholder with actual record ID
    - Stores stdout and stderr in context metadata
    - Fail-fast on non-zero exit codes
    - Respects timeout_seconds from config
    """

    command: str = Field(..., description="Shell command to execute (supports {record_id} placeholder)")
    timeout_seconds: float = Field(default=30.0, description="Timeout for shell command execution")

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Execute shell command and store output in context metadata.

        Args:
            context: Processing context containing the record to process

        Yields:
            The original record (output is stored in context.metadata)

        Raises:
            ValueError: If shell command fails (non-zero exit code)
            asyncio.TimeoutError: If command exceeds timeout_seconds
        """
        # Replace placeholders in command
        command = self.command.replace("{record_id}", context.record.record_id)

        logger.debug(
            "Executing shell command",
            record_id=context.record.record_id,
            command=command,
            timeout=self.timeout_seconds,
        )

        try:
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for completion with timeout
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            # Decode output
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

            # Store output in context metadata
            context.update_metadata("stdout", stdout)
            context.update_metadata("stderr", stderr)
            context.update_metadata("exit_code", process.returncode)

            # Fail-fast on non-zero exit code
            if process.returncode != 0:
                logger.error(
                    "Shell command failed",
                    record_id=context.record.record_id,
                    command=command,
                    exit_code=process.returncode,
                    stderr=stderr,
                )
                raise ValueError(f"Shell command failed with exit code {process.returncode}: {command}\nstderr: {stderr}")

            logger.debug(
                "Shell command completed successfully",
                record_id=context.record.record_id,
                stdout_length=len(stdout),
                stderr_length=len(stderr),
            )

        except asyncio.TimeoutError:
            logger.error(
                "Shell command timed out",
                record_id=context.record.record_id,
                command=command,
                timeout=self.timeout_seconds,
            )
            raise asyncio.TimeoutError(f"Shell command timed out after {self.timeout_seconds}s: {command}")
        except Exception as e:
            logger.error(
                "Error executing shell command",
                record_id=context.record.record_id,
                command=command,
                error=str(e),
            )
            raise

        # Yield the original record (output is stored in context metadata)
        yield context.record


class FilterProcessor(UnifiedProcessor):
    """Filters records based on JMESPath criteria.

    Evaluates a JMESPath expression against each record and yields the record
    only if the expression evaluates to a truthy value. Records that don't match
    are filtered out (nothing is yielded).

    Features:
    - Compiles JMESPath expression once during initialization for performance
    - Yields record if criteria evaluates to truthy (True, non-zero, non-empty)
    - Yields nothing if criteria evaluates to falsy (False, 0, None, empty)
    - Fail-fast on invalid JMESPath expressions

    Example criteria:
        - "metadata.status == 'active'" - Filter by equality
        - "length(metadata.tags) > `2`" - Filter by list length
        - "metadata.score >= `80`" - Filter by numeric threshold
    """

    criteria: str = Field(..., description="JMESPath expression for filtering")

    _compiled_criteria: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Compile JMESPath criteria after Pydantic initialization."""
        try:
            self._compiled_criteria = jmespath.compile(self.criteria)
        except JMESPathError as e:
            logger.error(
                "Invalid JMESPath criteria expression",
                criteria=self.criteria,
                error=str(e),
            )
            raise ValueError(f"Invalid JMESPath criteria: {self.criteria}") from e

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Filter record based on JMESPath criteria.

        Args:
            context: Processing context containing the record to filter

        Yields:
            The record if criteria evaluates to truthy, nothing otherwise

        Raises:
            ValueError: If JMESPath criteria evaluation fails
        """
        logger.debug(
            "Evaluating filter criteria",
            record_id=context.record.record_id,
            criteria=self.criteria,
        )

        # Convert record to dict for JMESPath processing
        record_dict = context.record.model_dump()

        try:
            # Evaluate JMESPath criteria
            result = self._compiled_criteria.search(record_dict)

            # Yield record only if criteria is truthy
            if result:
                logger.debug(
                    "Record passed filter criteria",
                    record_id=context.record.record_id,
                    criteria_result=result,
                )
                yield context.record
            else:
                logger.debug(
                    "Record filtered out by criteria",
                    record_id=context.record.record_id,
                    criteria_result=result,
                )
                # Yield nothing - record is filtered out

        except Exception as e:
            logger.error(
                "Error evaluating filter criteria",
                record_id=context.record.record_id,
                criteria=self.criteria,
                error=str(e),
            )
            raise ValueError(f"Error evaluating filter criteria '{self.criteria}': {str(e)}") from e


class EmbeddingProcessor(UnifiedBatchProcessor):
    """Batch processor for generating embeddings for document chunks.

    Processes batches of records with chunks, generating embeddings via Google GenAI.
    Implements retry logic with exponential backoff and semaphore-based concurrency control.

    Features:
    - Batches embedding API calls for efficiency
    - Retry logic with exponential backoff for rate limits
    - Semaphore-based concurrency control
    - Enriches context metadata with embedding statistics
    - Handles records without chunks gracefully
    """

    embedding_model: str = Field(..., description="Model identifier for embedding generation")
    dimensionality: int = Field(default=3072, description="Output embedding dimensionality")
    task: str = Field(default="RETRIEVAL_DOCUMENT", description="Task type for embedding model")
    embedding_max_retries: int = Field(default=5, description="Maximum retry attempts for embedding API")
    embedding_min_wait_seconds: float = Field(default=1.0, description="Minimum wait time for exponential backoff")
    embedding_max_wait_seconds: float = Field(default=120.0, description="Maximum wait time for exponential backoff")
    embedding_cooldown_seconds: float = Field(default=0.1, description="Cooldown between embedding batches")

    _embedding_semaphore: Any = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize special state attributes after Pydantic initialization."""
        self._embedding_semaphore = asyncio.Semaphore(20)
        self._client = None

    @property
    def client(self) -> genai.Client:
        """Lazily initialize the Gemini client on first access."""
        if self._client is None:
            try:
                from buttermilk import bm

                self._client = bm.genai
            except (RuntimeError, AttributeError):
                # For testing, create a client directly
                self._client = genai.Client()
        return self._client

    async def _process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> AsyncGenerator[list[BaseRecord], None]:
        """Process batch of contexts by generating embeddings for their chunks.

        Args:
            contexts: List of ProcessingContext objects to process

        Yields:
            list[BaseRecord]: Batch of records with embeddings added to chunks
        """
        start_time = time.time()

        # Collect all records and their chunks
        records_with_chunks = []
        for context in contexts:
            if hasattr(context.record, "chunks") and context.record.chunks:
                records_with_chunks.append(context)
            else:
                logger.debug(
                    "Record has no chunks to embed, skipping",
                    record_id=context.record.record_id,
                )

        if not records_with_chunks:
            # No records with chunks, yield all records unchanged
            logger.debug("No records with chunks in batch")
            yield [ctx.record for ctx in contexts]
            return

        # Generate embeddings for all chunks across all records
        success = await self._embed_all_chunks(records_with_chunks)

        processing_time_ms = (time.time() - start_time) * 1000

        # Update context metadata for all successful records
        for context in records_with_chunks:
            context.update_metadata(
                "embedding_stats",
                {
                    "chunks_embedded": len(context.record.chunks),
                    "embedding_model": self.embedding_model,
                    "processing_time_ms": processing_time_ms,
                },
            )

        logger.info(
            "Successfully generated embeddings for batch",
            batch_size=len(records_with_chunks),
            total_chunks=sum(len(ctx.record.chunks) for ctx in records_with_chunks),
            processing_time_ms=processing_time_ms,
        )

        # Yield all records (including those without chunks)
        yield [ctx.record for ctx in contexts]

    async def _embed_all_chunks(self, contexts: list[ProcessingContext]) -> None:
        """Generate embeddings for all chunks across all contexts.

        Args:
            contexts: List of contexts with records that have chunks

        Raises:
            ValueError: If no chunks found or embeddings fail
        """
        # Build list of (context_idx, chunk_idx, text) tuples
        embeddings_input = []
        for ctx_idx, context in enumerate(contexts):
            for chunk_idx, chunk in enumerate(context.record.chunks):
                # Support both dict and object chunks
                if isinstance(chunk, dict):
                    text = chunk.get("text", "")
                elif hasattr(chunk, "text"):
                    text = chunk.text
                else:
                    logger.warning(f"Unsupported chunk type: {type(chunk)}")
                    continue

                embeddings_input.append((ctx_idx, chunk_idx, text))

        if not embeddings_input:
            raise ValueError("No chunks found to embed")

        # Generate embeddings with retry logic
        embedding_results = await self._embed(embeddings_input)

        # Apply embeddings back to chunks
        success_count = 0
        for ctx_idx, chunk_idx, embedding in embedding_results:
            if embedding is not None:
                context = contexts[ctx_idx]
                chunk = context.record.chunks[chunk_idx]

                # Set embedding based on chunk type
                if isinstance(chunk, dict):
                    chunk["embedding"] = embedding
                elif hasattr(chunk, "embedding"):
                    chunk.embedding = embedding

                success_count += 1

        total_chunks = len(embeddings_input)
        if success_count == 0:
            logger.error("All embeddings failed")
            raise ValueError("All embeddings failed")

        if success_count < total_chunks:
            logger.warning(
                "Partial embedding failure",
                succeeded=success_count,
                total=total_chunks,
            )
            # Clear embeddings to avoid partial state
            for context in contexts:
                for chunk in context.record.chunks:
                    if isinstance(chunk, dict):
                        chunk["embedding"] = None
                    elif hasattr(chunk, "embedding"):
                        chunk.embedding = None
            raise ValueError(f"Partial embedding failure: {success_count}/{total_chunks} succeeded")

        logger.debug("Generated embeddings", count=success_count)

    async def _embed(
        self,
        embeddings_input: list[tuple[int, int, str]],
    ) -> list[tuple[int, int, list[float] | None]]:
        """Generate embeddings with retry logic.

        Args:
            embeddings_input: List of (context_idx, chunk_idx, text) tuples

        Returns:
            List of (context_idx, chunk_idx, embedding) tuples where embedding can be None on failure
        """

        async def _run_embed_batch(batch_texts: list[str], attempt: int = 0):
            """Run embedding for a batch with semaphore."""
            async with self._embedding_semaphore:
                try:
                    response = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=batch_texts,
                        config={
                            "output_dimensionality": self.dimensionality,
                            "auto_truncate": False,
                        },
                    )

                    # Extract embeddings from response
                    embeddings = []
                    for embedding in response.embeddings:
                        embeddings.append(scrub_serializable(embedding.values))

                    logger.debug(
                        "Embedding batch",
                        batch_size=len(batch_texts),
                        attempt=attempt + 1,
                        embeddings_count=len(embeddings),
                    )

                    # Add cooldown to avoid rate limits
                    if self.embedding_cooldown_seconds > 0:
                        await asyncio.sleep(self.embedding_cooldown_seconds)

                    return embeddings

                except Exception as e:
                    logger.exception(
                        "Embedding API error",
                        error=str(e),
                        batch_size=len(batch_texts),
                        attempt=attempt + 1,
                    )
                    raise

        # Process in batches
        results: list[tuple[int, int, list[float] | None]] = []
        batch_size = self.batch_size

        for i in range(0, len(embeddings_input), batch_size):
            batch = embeddings_input[i : i + batch_size]
            batch_metadata = [(ctx_idx, chunk_idx) for ctx_idx, chunk_idx, _ in batch]
            batch_texts = [text for _, _, text in batch]

            # Retry logic
            last_exception = None
            for attempt in range(self.embedding_max_retries):
                try:
                    embeddings = await _run_embed_batch(batch_texts, attempt=attempt)

                    # Pair metadata with embeddings
                    for (ctx_idx, chunk_idx), embedding in zip(batch_metadata, embeddings):
                        results.append((ctx_idx, chunk_idx, embedding))
                    break

                except Exception as exc:
                    last_exception = exc
                    if self._is_rate_limit_error(exc):
                        # Exponential backoff for rate limits
                        wait_time = min(
                            self.embedding_min_wait_seconds * (2**attempt),
                            self.embedding_max_wait_seconds,
                        )
                        logger.warning(
                            "Rate limit hit, retrying",
                            attempt=attempt + 1,
                            wait_time=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                    elif attempt == self.embedding_max_retries - 1:
                        # Last attempt failed
                        logger.error(
                            "Embedding batch failed after retries",
                            batch_size=len(batch_texts),
                            error=str(exc),
                        )
                        # Re-raise the original exception for fail-fast behavior
                        raise
                    else:
                        # Non-rate limit error, retry immediately
                        logger.warning(
                            "Embedding error, retrying",
                            attempt=attempt + 1,
                            error=str(exc),
                        )

        return results

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        msg = str(exc).lower()
        return any(k in msg for k in ["rate limit", "quota", "too many requests", "429"])


class ChromaDBProcessor(UnifiedProcessor):
    """Upload records with embeddings to ChromaDB.

    Processes records with embedded chunks and uploads them to a ChromaDB collection.
    Handles remote storage, caching, and syncing to remote backends.

    This processor is typically used for batch updates to vector databases and
    wouldn't be included in production RAG pipelines.

    Features:
    - Processes single records with chunks and embeddings
    - Uploads to ChromaDB collection via upserts
    - Handles remote storage paths (gs://, s3://, etc)
    - Automatic syncing based on batch size or time interval
    - Final sync via finalize_processing()
    - Enriches context metadata with upload statistics
    """

    collection_name: str = Field(..., description="ChromaDB collection name")
    persist_directory: str = Field(..., description="ChromaDB persistence directory path")
    sync_batch_size: int = Field(default=50, description="Number of records processed before sync")
    sync_interval_minutes: int = Field(default=10, description="Time interval between syncs in minutes")
    disable_auto_sync: bool = Field(default=False, description="Disable automatic syncing to remote")
    upsert_batch_size: int = Field(default=1000, description="Batch size for ChromaDB upserts")

    _client: Any = PrivateAttr(default=None)
    _collection: Any = PrivateAttr(default=None)
    _original_remote_path: Any = PrivateAttr(default=None)
    _processed_count: int = PrivateAttr(default=0)
    _last_sync_time: float = PrivateAttr(default=0.0)
    _cache_initialized: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        """Initialize special state attributes after Pydantic initialization."""
        self._client = None
        self._collection = None
        self._original_remote_path = None
        self._processed_count = 0
        self._last_sync_time = time.time()
        self._cache_initialized = False

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by uploading its embedded chunks to ChromaDB.

        Args:
            context: Processing context containing the record to process

        Yields:
            BaseRecord: The original record with updated metadata (passthrough after upload)
        """
        # Ensure cache is initialized for remote storage
        if not self._cache_initialized:
            await self._ensure_cache_initialized()

        # Check if record has embedded chunks
        chunks_count = len(getattr(context.record, "chunks", []))
        logger.debug(
            "ChromaDBProcessor received record",
            record_id=context.record.record_id,
            has_chunks=hasattr(context.record, "chunks"),
            chunks_count=chunks_count,
        )

        if not hasattr(context.record, "chunks") or not context.record.chunks:
            logger.warning(
                "Record has no chunks to upload",
                record_id=context.record.record_id,
            )
            yield context.record
            return

        # Check if chunks have embeddings
        chunks_with_embeddings = [
            c for c in context.record.chunks if (c.get("embedding") if isinstance(c, dict) else getattr(c, "embedding", None)) is not None
        ]
        logger.debug(
            "ChromaDBProcessor chunk embedding status",
            record_id=context.record.record_id,
            total_chunks=len(context.record.chunks),
            chunks_with_embeddings=len(chunks_with_embeddings),
        )

        if not chunks_with_embeddings:
            logger.warning(
                "Record chunks have no embeddings",
                record_id=context.record.record_id,
                chunks_count=len(context.record.chunks),
            )
            yield context.record
            return

        # Upload chunks to ChromaDB
        start_time = time.time()
        try:
            await self._store_chunks_for_record(context.record)
            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Successfully uploaded to ChromaDB",
                record_id=context.record.record_id,
                chunks_uploaded=len(chunks_with_embeddings),
                processing_time_ms=processing_time_ms,
            )

            # Update processed count
            self._processed_count += 1

            # Check if we need to sync
            await self._maybe_sync()

            # Update context metadata with statistics
            context.update_metadata(
                "chromadb_stats",
                {
                    "chunks_uploaded": len(chunks_with_embeddings),
                    "collection_name": self.collection_name,
                    "processing_time_ms": processing_time_ms,
                },
            )

            # Add metadata to record about upload
            metadata = context.record.metadata.copy() if context.record.metadata else {}
            metadata["chromadb_upload"] = {
                "status": "uploaded",
                "timestamp": time.time(),
                "processor": "ChromaDBProcessor",
                "chunks_uploaded": len(chunks_with_embeddings),
                "collection": self.collection_name,
                "processing_time_ms": processing_time_ms,
            }

            # Yield record with updated metadata
            processed_record = context.record.model_copy(update={"metadata": metadata})
            yield processed_record

        except Exception as e:
            logger.error(
                "Failed to upload to ChromaDB",
                record_id=context.record.record_id,
                error=str(e),
            )
            # Re-raise for fail-fast behavior
            raise

    async def _ensure_cache_initialized(self) -> None:
        """Ensure ChromaDB cache and collection are ready for use.

        Raises:
            ValueError: If collection initialization fails
        """
        persist_dir = self.persist_directory

        # Handle remote storage by downloading to local cache
        if persist_dir.startswith(("gs://", "s3://", "azure://", "gcs://")):
            self._original_remote_path = persist_dir
            local_cache_path = await self._setup_local_cache(persist_dir)
            persist_dir = str(local_cache_path)

        # Initialize ChromaDB client
        if not self._client:
            self._client = await asyncio.to_thread(
                chromadb.PersistentClient,
                path=persist_dir,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            logger.info("ChromaDB client initialized", persist_directory=persist_dir)

        # Get or create collection
        if not self._collection:
            try:
                self._collection = await asyncio.to_thread(self._client.get_collection, name=self.collection_name)
                collection_count = await asyncio.to_thread(self._collection.count)
                logger.info(
                    "Using existing collection",
                    collection_name=self.collection_name,
                    count=collection_count,
                )
            except Exception:
                self._collection = await asyncio.to_thread(self._client.create_collection, name=self.collection_name)
                logger.info("Created new collection", collection_name=self.collection_name)

        self._cache_initialized = True

    async def _setup_local_cache(self, remote_path: str) -> Path:
        """Setup local cache for remote ChromaDB.

        Args:
            remote_path: Remote storage path (e.g., gs://bucket/path)

        Returns:
            Path: Local cache directory path
        """
        # Use SessionInfo for consistent cache key generation
        cache_key = bm.session_info.generate_cache_key(remote_path)
        local_cache_path = bm.session_info.get_chromadb_cache_dir() / cache_key
        local_cache_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Using local cache for remote ChromaDB",
            remote_path=remote_path,
            local_cache=str(local_cache_path),
        )

        return local_cache_path

    async def _store_chunks_for_record(self, record: BaseRecord) -> None:
        """Store record chunks with metadata in ChromaDB.

        Args:
            record: BaseRecord with chunks and embeddings

        Raises:
            ValueError: If collection not initialized or no chunks to store
        """
        if not self._collection:
            raise ValueError("Collection not initialized")

        chunks_to_upsert = []

        # Convert chunks to dicts if needed and filter for embeddings
        for c in record.chunks:
            if hasattr(c, "model_dump"):
                # Convert ChunkedDocument to dict
                chunk_dict = c.model_dump()
            elif isinstance(c, dict):
                chunk_dict = c
            else:
                # Skip unsupported chunk types
                continue

            if chunk_dict.get("embedding") is not None:
                chunks_to_upsert.append(chunk_dict)

        if not chunks_to_upsert:
            return

        ids = []
        documents = []
        embeddings_list = []
        metadatas = []

        for chunk in chunks_to_upsert:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["chunk_text"])

            # Convert numpy array or list to regular Python floats
            embedding = chunk["embedding"]
            if hasattr(embedding, "tolist"):
                embeddings_list.append(embedding.tolist())
            else:
                embeddings_list.append([float(x) for x in embedding])

            # Build metadata
            chunk_metadata = chunk.get("metadata", {})
            enhanced_metadata = {
                "document_title": chunk["document_title"],
                "chunk_index": chunk["chunk_index"],
                "document_id": chunk["document_id"],
                "content_type": chunk_metadata.get("content_type", "unknown"),
                "chunk_type": chunk_metadata.get("chunk_type", "unknown"),
                **{k: v for k, v in chunk_metadata.items() if k not in ["content_type", "chunk_type"]},
            }
            # Ensure metadata is serializable and ChromaDB-compatible
            enhanced_metadata = scrub_serializable(enhanced_metadata)
            metadatas.append(_sanitize_metadata_for_chroma(enhanced_metadata))

        # Batch upsert to ChromaDB
        for i in range(0, len(ids), self.upsert_batch_size):
            batch_end = min(i + self.upsert_batch_size, len(ids))

            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

            logger.debug(
                "Upserted batch to ChromaDB",
                record_id=record.record_id,
                batch_start=i,
                batch_end=batch_end,
                total=len(ids),
            )

    async def _maybe_sync(self) -> None:
        """Sync to remote if conditions are met."""
        if self.disable_auto_sync:
            return

        # Check if we should sync based on count or time
        should_sync = False
        current_time = time.time()

        if self._processed_count >= self.sync_batch_size:
            should_sync = True
            reason = f"batch size ({self._processed_count} records)"
        elif (current_time - self._last_sync_time) >= (self.sync_interval_minutes * 60):
            should_sync = True
            reason = f"time interval ({self.sync_interval_minutes} minutes)"

        if should_sync and self._original_remote_path:
            logger.debug(
                "Syncing to remote storage",
                reason=reason,
                processed_count=self._processed_count,
            )
            try:
                # Get the local cache path
                local_cache_path = await self._get_local_cache_path()
                if local_cache_path:
                    await upload_chromadb_cache(str(local_cache_path), self._original_remote_path)
                    logger.info(
                        "Successfully synced ChromaDB to remote storage",
                        processed_count=self._processed_count,
                    )
                else:
                    logger.warning("Could not determine local cache path for sync")
            except Exception as e:
                logger.error(
                    "Failed to sync to remote storage",
                    error=str(e),
                    processed_count=self._processed_count,
                )
            finally:
                # Reset counters regardless of sync success/failure
                self._processed_count = 0
                self._last_sync_time = current_time

    async def finalize_processing(self) -> bool:
        """Finalize processing by syncing to remote.

        Returns:
            bool: True if finalization successful, False otherwise
        """
        if self._original_remote_path:
            logger.info("Final sync to remote storage", processed_count=self._processed_count)
            try:
                # Get the local cache path
                local_cache_path = await self._get_local_cache_path()
                if local_cache_path:
                    await upload_chromadb_cache(str(local_cache_path), self._original_remote_path)
                    logger.info(
                        "Successfully completed final sync to remote storage",
                        processed_count=self._processed_count,
                    )
                    return True
                else:
                    logger.warning("Could not determine local cache path for final sync")
                    return False
            except Exception as e:
                logger.error(
                    "Failed final sync to remote storage",
                    error=str(e),
                    processed_count=self._processed_count,
                )
                return False
        return True

    async def _get_local_cache_path(self) -> Path | None:
        """Get the local cache path for the ChromaDB instance.

        Returns:
            Path: Local cache path if exists, None otherwise
        """
        if not self._original_remote_path:
            return None

        # Recreate the cache path logic from _setup_local_cache (must match SessionInfo)
        cache_key = bm.session_info.generate_cache_key(self._original_remote_path)
        local_cache_path = bm.session_info.get_chromadb_cache_dir() / cache_key

        if local_cache_path.exists():
            return local_cache_path
        else:
            logger.warning("Local cache path does not exist", cache_path=str(local_cache_path))
            return None
