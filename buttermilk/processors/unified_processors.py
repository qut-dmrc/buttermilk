"""Standard Unified Processors implementations.

This module contains concrete implementations of standard processors:
- GroupchatProcessor: Runs a group chat session.
- ExpanderProcessor: Expands a record into multiple records (1:N).
- TransformProcessor: Applies JMESPath transformations to records.
- ShellProcessor: Executes shell commands with placeholder substitution.
- FilterProcessor: Filters records based on JMESPath criteria.
- EmbeddingProcessor: Generates embeddings for chunks in batch.

Processors are automatically registered on import.
"""

import asyncio
import time
from typing import Any, AsyncGenerator

import jmespath
from jmespath.exceptions import JMESPathError
from google import genai

from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import (
    EmbeddingProcessorConfig,
    ExpanderProcessorConfig,
    FilterProcessorConfig,
    GroupchatProcessorConfig,
    ShellProcessorConfig,
    TransformProcessorConfig,
)
from buttermilk._core.processor_registry import register_processor
from buttermilk._core.types import BaseRecord, RunRequest
from buttermilk._core.unified_processor import UnifiedProcessor
from buttermilk._core.unified_batch_processor import UnifiedBatchProcessor
from buttermilk.runner.flowrunner import OrchestratorFactory
from buttermilk import logger
from buttermilk.utils.utils import scrub_serializable


class GroupchatProcessor(UnifiedProcessor):
    """Wraps an Orchestrator to run multi-agent conversations as a unified processor.

    Creates a fresh orchestrator instance per record to ensure state isolation.
    Collects ExecutionTrace outputs via callback and enriches record metadata with results.

    This processor enables complex multi-agent flows to be used within the unified
    processor architecture while maintaining proper isolation and observability.
    """

    def __init__(self, config: GroupchatProcessorConfig):
        super().__init__(config)
        self.config: GroupchatProcessorConfig = config

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
            flow_name=self.config.flow_name,
        )

        # Create fresh orchestrator for each record (state isolation)
        orchestrator = OrchestratorFactory.create_orchestrator(
            self.config.flow_config, self.config.flow_name
        )

        # Collect ExecutionTrace outputs via callback
        traces: list[ExecutionTrace] = []

        async def collect_callback(message: Any) -> None:
            """Callback to collect ExecutionTrace outputs from orchestrator."""
            if isinstance(message, ExecutionTrace):
                traces.append(message)

        # Use ui_callback from context if available, otherwise use collect_callback
        callback = context.ui_callback if context.ui_callback else collect_callback

        # If context has ui_callback but we need to collect traces, chain callbacks
        if context.ui_callback and self.config.collect_traces:
            async def chained_callback(message: Any) -> None:
                """Chain both callbacks."""
                await context.ui_callback(message)
                await collect_callback(message)
            callback = chained_callback
        elif self.config.collect_traces:
            callback = collect_callback
        else:
            callback = context.ui_callback

        # Create RunRequest from record
        run_request = RunRequest(
            flow=self.config.flow_name,
            inputs={
                "record_id": record_id,
                "record": context.record.model_dump() if hasattr(context.record, "model_dump") else context.record,
            },
            parameters=self.config.parameters,
            callback_to_ui=callback,
        )

        try:
            # Run orchestrator (returns None, results flow through callback)
            await orchestrator.run(request=run_request)

            # Build outputs from collected traces
            outputs = []
            for trace in traces:
                if trace.outputs is not None:
                    outputs.append(trace.outputs)

            # Enrich record metadata with orchestrator results
            enriched_metadata = {
                **(context.record.metadata if context.record.metadata else {}),
                "groupchat": {
                    "status": "processed",
                    "flow_name": self.config.flow_name,
                    "trace_count": len(traces),
                    "outputs": outputs,
                },
            }

            logger.debug(
                "GroupchatProcessor completed",
                record_id=record_id,
                flow_name=self.config.flow_name,
                trace_count=len(traces),
                output_count=len(outputs),
            )

            yield context.record.model_copy(update={"metadata": enriched_metadata})

        except Exception as e:
            logger.error(
                "GroupchatProcessor failed",
                record_id=record_id,
                flow_name=self.config.flow_name,
                error=str(e),
            )
            # Re-raise to let pipeline handle the error
            raise


class ExpanderProcessor(UnifiedProcessor):
    """Expands a single record into multiple records based on a list field."""
    
    def __init__(self, config: ExpanderProcessorConfig):
        super().__init__(config)
        self.config: ExpanderProcessorConfig = config

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        field_name = self.config.field_to_expand
        
        # Access the field from the record (attribute or metadata)
        values = getattr(context.record, field_name, None)
        if values is None:
            values = context.record.metadata.get(field_name)
            
        if not isinstance(values, list):
            logger.warning(
                f"Field '{field_name}' is not a list, cannot expand. Skipping expansion.",
                record_id=context.record.record_id,
                value_type=type(values)
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

    def __init__(self, config: TransformProcessorConfig):
        super().__init__(config)
        self.config: TransformProcessorConfig = config

        # Compile JMESPath expression once during initialization
        try:
            self._compiled_expression = jmespath.compile(config.expression)
        except JMESPathError as e:
            logger.error(
                "Invalid JMESPath expression",
                expression=config.expression,
                error=str(e),
            )
            raise ValueError(
                f"Invalid JMESPath expression: {config.expression}"
            ) from e

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
            expression=self.config.expression,
            output_field=self.config.output_field,
        )

        # Convert record to dict for JMESPath processing
        record_dict = context.record.model_dump()

        try:
            # Apply JMESPath expression
            result = self._compiled_expression.search(record_dict)

            if result is not None:
                # Store result in context metadata
                context.update_metadata(self.config.output_field, result)

                logger.debug(
                    "JMESPath transformation complete",
                    record_id=context.record.record_id,
                    output_field=self.config.output_field,
                    result_type=type(result).__name__,
                )
            else:
                logger.debug(
                    "JMESPath expression returned None, no metadata stored",
                    record_id=context.record.record_id,
                    expression=self.config.expression,
                )

        except Exception as e:
            logger.error(
                "Error applying JMESPath expression",
                record_id=context.record.record_id,
                expression=self.config.expression,
                error=str(e),
            )
            raise ValueError(
                f"Error applying JMESPath expression '{self.config.expression}': {str(e)}"
            ) from e

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

    def __init__(self, config: ShellProcessorConfig):
        super().__init__(config)
        self.config: ShellProcessorConfig = config

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
        command = self.config.command.replace("{record_id}", context.record.record_id)

        logger.debug(
            "Executing shell command",
            record_id=context.record.record_id,
            command=command,
            timeout=self.config.timeout_seconds,
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
                timeout=self.config.timeout_seconds,
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
                raise ValueError(
                    f"Shell command failed with exit code {process.returncode}: {command}\n"
                    f"stderr: {stderr}"
                )

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
                timeout=self.config.timeout_seconds,
            )
            raise asyncio.TimeoutError(
                f"Shell command timed out after {self.config.timeout_seconds}s: {command}"
            )
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

    def __init__(self, config: FilterProcessorConfig):
        super().__init__(config)
        self.config: FilterProcessorConfig = config

        # Compile JMESPath expression once during initialization
        try:
            self._compiled_criteria = jmespath.compile(config.criteria)
        except JMESPathError as e:
            logger.error(
                "Invalid JMESPath criteria expression",
                criteria=config.criteria,
                error=str(e),
            )
            raise ValueError(
                f"Invalid JMESPath criteria: {config.criteria}"
            ) from e

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
            criteria=self.config.criteria,
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
                criteria=self.config.criteria,
                error=str(e),
            )
            raise ValueError(
                f"Error evaluating filter criteria '{self.config.criteria}': {str(e)}"
            ) from e


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

    def __init__(self, config: EmbeddingProcessorConfig):
        super().__init__(config)
        self.config: EmbeddingProcessorConfig = config
        self._embedding_semaphore = asyncio.Semaphore(20)
        self._client: genai.Client | None = None

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
            context.update_metadata("embedding_stats", {
                "chunks_embedded": len(context.record.chunks),
                "embedding_model": self.config.embedding_model,
                "processing_time_ms": processing_time_ms,
            })

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
        for (ctx_idx, chunk_idx, embedding) in embedding_results:
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
                        model=self.config.embedding_model,
                        contents=batch_texts,
                        config={
                            "output_dimensionality": self.config.dimensionality,
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
                    if self.config.embedding_cooldown_seconds > 0:
                        await asyncio.sleep(self.config.embedding_cooldown_seconds)

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
        batch_size = self.config.batch_size

        for i in range(0, len(embeddings_input), batch_size):
            batch = embeddings_input[i : i + batch_size]
            batch_metadata = [(ctx_idx, chunk_idx) for ctx_idx, chunk_idx, _ in batch]
            batch_texts = [text for _, _, text in batch]

            # Retry logic
            last_exception = None
            for attempt in range(self.config.embedding_max_retries):
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
                            self.config.embedding_min_wait_seconds * (2**attempt),
                            self.config.embedding_max_wait_seconds,
                        )
                        logger.warning(
                            "Rate limit hit, retrying",
                            attempt=attempt + 1,
                            wait_time=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                    elif attempt == self.config.embedding_max_retries - 1:
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
        return any(
            k in msg for k in ["rate limit", "quota", "too many requests", "429"]
        )


# Register processors on module import
register_processor("groupchat", GroupchatProcessor)
register_processor("expander", ExpanderProcessor)
register_processor("transform", TransformProcessor)
register_processor("shell", ShellProcessor)
register_processor("filter", FilterProcessor)
register_processor("embedding", EmbeddingProcessor)
