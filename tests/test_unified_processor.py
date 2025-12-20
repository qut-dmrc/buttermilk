"""Tests for the Unified Processor Architecture.

This module tests the core components of the new unified processor system:
- ProcessingContext: State container for processor chains
- UnifiedProcessor: Base class with tracing and error handling
- ExpanderProcessor: 1:N record expansion
- PipelineExecutor: Processor chain orchestration

Tests use REAL data patterns (no mocking internal code) and follow fail-fast philosophy.
"""

from typing import AsyncGenerator

import pytest
from opentelemetry import trace

from buttermilk._core.executor import PipelineExecutor
from buttermilk._core.pipeline_config import PipelineConfig
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import BatchProcessorConfig, ExpanderProcessorConfig, ProcessorConfig, TransformProcessorConfig
from buttermilk._core.protocols import BatchProcessor, Processor
from buttermilk._core.types import BaseRecord
from buttermilk._core.unified_processor import UnifiedProcessor
from buttermilk._core.unified_batch_processor import UnifiedBatchProcessor
from buttermilk.processors.unified_processors import ExpanderProcessor, TransformProcessor


class TestProcessingContext:
    """Test ProcessingContext creation and manipulation."""

    def test_processing_context_creation(self):
        """Verify ProcessingContext can be created with required fields."""
        # Create a minimal BaseRecord
        record = BaseRecord(
            record_id="test-123",
            content="test content",
        )

        # Create ProcessingContext with required fields
        context = ProcessingContext(
            session_id="session-abc",
            record=record,
        )

        # Verify required fields are set
        assert context.session_id == "session-abc"
        assert context.record.record_id == "test-123"
        assert context.record.content == "test content"

        # Verify optional fields have correct defaults
        assert context.batch_id is None
        assert context.metadata == {}
        assert context.span is None
        assert context.ui_callback is None
        assert context.resources == {}

    def test_processing_context_with_optional_fields(self):
        """Verify ProcessingContext can be created with optional fields."""
        record = BaseRecord(record_id="test-456", content="more content")

        # Create context with optional fields
        context = ProcessingContext(
            session_id="session-xyz",
            record=record,
            batch_id="batch-001",
            metadata={"stage": "initial", "attempt": 1},
            resources={"db": "mock_connection"},
        )

        assert context.batch_id == "batch-001"
        assert context.metadata == {"stage": "initial", "attempt": 1}
        assert context.resources == {"db": "mock_connection"}

    def test_processing_context_metadata_update(self):
        """Verify metadata can be updated."""
        record = BaseRecord(record_id="test-789", content="content")
        context = ProcessingContext(
            session_id="session-update",
            record=record,
        )

        # Initially empty
        assert context.metadata == {}

        # Update metadata
        context.update_metadata("key1", "value1")
        assert context.metadata == {"key1": "value1"}

        # Update with more values
        context.update_metadata("key2", 42)
        context.update_metadata("key3", {"nested": "data"})

        assert context.metadata == {
            "key1": "value1",
            "key2": 42,
            "key3": {"nested": "data"},
        }

        # Updating existing key overwrites
        context.update_metadata("key1", "new_value")
        assert context.metadata["key1"] == "new_value"

    def test_processing_context_get_resource(self):
        """Verify resources can be retrieved."""
        record = BaseRecord(record_id="test-res", content="content")
        context = ProcessingContext(
            session_id="session-res",
            record=record,
            resources={"db": "connection", "api": "client"},
        )

        # Get existing resource
        assert context.get_resource("db") == "connection"
        assert context.get_resource("api") == "client"

        # Get non-existent resource raises KeyError (fail-fast)
        with pytest.raises(KeyError, match="nonexistent"):
            context.get_resource("nonexistent")


class TestExpanderProcessor:
    """Test ExpanderProcessor for 1:N record expansion."""

    @pytest.mark.anyio
    async def test_expander_processor_expands_list_field(self):
        """Verify ExpanderProcessor yields multiple records from a list field."""
        # Create config to expand a list field in metadata
        config = ExpanderProcessorConfig(
            type="expander",
            field_to_expand="items",
        )

        processor = ExpanderProcessor(config)

        # Create a record with a list in metadata
        record = BaseRecord(
            record_id="parent-001",
            content="parent content",
            metadata={
                "items": ["item_a", "item_b", "item_c"],
                "source": "test",
            },
        )

        context = ProcessingContext(
            session_id="session-expand",
            record=record,
        )

        # Collect all outputs
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield 3 records (one per item)
        assert len(outputs) == 3

        # Verify each output has correct properties
        for i, output_record in enumerate(outputs):
            # Record ID should be suffixed with index
            assert output_record.record_id == f"parent-001_{i}"

            # Metadata should contain expansion info
            assert output_record.metadata["expansion_source_id"] == "parent-001"
            assert output_record.metadata["expansion_index"] == i

            # Field value should be the individual item
            assert output_record.metadata["items"] in ["item_a", "item_b", "item_c"]

            # Original metadata should be preserved
            assert output_record.metadata["source"] == "test"

    @pytest.mark.anyio
    async def test_expander_processor_passthrough_non_list(self):
        """Verify non-list field passes through unchanged."""
        config = ExpanderProcessorConfig(
            type="expander",
            field_to_expand="not_a_list",
        )

        processor = ExpanderProcessor(config)

        # Create record where field is not a list
        record = BaseRecord(
            record_id="single-001",
            content="single content",
            metadata={"not_a_list": "just a string"},
        )

        context = ProcessingContext(
            session_id="session-passthrough",
            record=record,
        )

        # Collect outputs
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield exactly 1 record (passthrough)
        assert len(outputs) == 1

        # Record should be unchanged
        output_record = outputs[0]
        assert output_record.record_id == "single-001"
        assert output_record.content == "single content"
        assert output_record.metadata["not_a_list"] == "just a string"

    @pytest.mark.anyio
    async def test_expander_processor_missing_field(self):
        """Verify processor handles missing field gracefully."""
        config = ExpanderProcessorConfig(
            type="expander",
            field_to_expand="missing_field",
        )

        processor = ExpanderProcessor(config)

        record = BaseRecord(
            record_id="no-field-001",
            content="content",
            metadata={"other": "data"},
        )

        context = ProcessingContext(
            session_id="session-missing",
            record=record,
        )

        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should passthrough when field is missing
        assert len(outputs) == 1
        assert outputs[0].record_id == "no-field-001"


class TestTransformProcessor:
    """Test TransformProcessor for JMESPath transformations."""

    @pytest.mark.anyio
    async def test_transform_processor_applies_jmespath_expression(self):
        """Verify TransformProcessor evaluates JMESPath expression and stores result."""
        # Create config with a simple JMESPath expression
        config = TransformProcessorConfig(
            type="transform",
            expression="metadata.tags[0]",
            output_field="first_tag",
        )

        processor = TransformProcessor(config)

        # Create a record with nested data
        record = BaseRecord(
            record_id="transform-001",
            content="test content",
            metadata={
                "tags": ["python", "testing", "async"],
                "source": "test",
            },
        )

        context = ProcessingContext(
            session_id="session-transform",
            record=record,
        )

        # Process the record
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield exactly 1 record (the original)
        assert len(outputs) == 1
        assert outputs[0].record_id == "transform-001"

        # Result should be stored in context metadata
        assert context.metadata["first_tag"] == "python"

    @pytest.mark.anyio
    async def test_transform_processor_stores_in_output_field(self):
        """Verify TransformProcessor uses custom output_field for storing result."""
        # Create config with custom output field
        config = TransformProcessorConfig(
            type="transform",
            expression="content",
            output_field="extracted_content",
        )

        processor = TransformProcessor(config)

        record = BaseRecord(
            record_id="transform-002",
            content="Hello, World!",
            metadata={},
        )

        context = ProcessingContext(
            session_id="session-custom-field",
            record=record,
        )

        # Process the record
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        assert len(outputs) == 1

        # Check custom output field
        assert context.metadata["extracted_content"] == "Hello, World!"
        # Default field should not exist
        assert "transformed" not in context.metadata

    @pytest.mark.anyio
    async def test_transform_processor_complex_expression(self):
        """Verify TransformProcessor handles complex JMESPath expressions."""
        # Create config with complex expression
        config = TransformProcessorConfig(
            type="transform",
            expression="metadata.{name: author, tag_count: length(tags)}",
            output_field="summary",
        )

        processor = TransformProcessor(config)

        record = BaseRecord(
            record_id="transform-003",
            content="content",
            metadata={
                "author": "Alice",
                "tags": ["a", "b", "c"],
            },
        )

        context = ProcessingContext(
            session_id="session-complex",
            record=record,
        )

        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        assert len(outputs) == 1

        # Verify complex object result
        assert context.metadata["summary"] == {
            "name": "Alice",
            "tag_count": 3,
        }

    @pytest.mark.anyio
    async def test_transform_processor_expression_returns_none(self):
        """Verify processor handles None result gracefully (no metadata stored)."""
        config = TransformProcessorConfig(
            type="transform",
            expression="metadata.nonexistent_field",
            output_field="result",
        )

        processor = TransformProcessor(config)

        record = BaseRecord(
            record_id="transform-004",
            content="content",
            metadata={"other": "data"},
        )

        context = ProcessingContext(
            session_id="session-none",
            record=record,
        )

        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        assert len(outputs) == 1

        # No metadata should be stored when expression returns None
        assert "result" not in context.metadata

    @pytest.mark.anyio
    async def test_transform_processor_invalid_expression_raises(self):
        """Verify processor fails fast on invalid JMESPath expression."""
        # Invalid JMESPath syntax should raise during processor initialization
        config = TransformProcessorConfig(
            type="transform",
            expression="metadata..invalid[[syntax",
            output_field="result",
        )

        # Creating the processor should fail fast on invalid expression
        with pytest.raises(ValueError, match="Invalid JMESPath expression"):
            processor = TransformProcessor(config)


class TestPipelineExecutor:
    """Test PipelineExecutor for chaining processors."""

    @pytest.mark.anyio
    async def test_pipeline_executor_chains_processors(self):
        """Verify PipelineExecutor runs processors in sequence."""
        # Create a simple pipeline with one expander
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            processors=[
                ExpanderProcessorConfig(
                    type="expander",
                    field_to_expand="tags",
                )
            ],
        )

        executor = PipelineExecutor(pipeline_config)

        # Create source generator
        async def source() -> AsyncGenerator[BaseRecord, None]:
            yield BaseRecord(
                record_id="source-001",
                content="test content",
                metadata={"tags": ["python", "testing", "async"]},
            )

        # Run pipeline
        results = []
        async for result in executor.run(source(), session_id="test-session"):
            results.append(result)

        # Should produce 3 records (one per tag)
        assert len(results) == 3

        # Verify expansion occurred
        record_ids = [r.record_id for r in results]
        assert "source-001_0" in record_ids
        assert "source-001_1" in record_ids
        assert "source-001_2" in record_ids

    @pytest.mark.anyio
    async def test_pipeline_executor_multiple_source_records(self):
        """Verify executor processes multiple source records."""
        pipeline_config = PipelineConfig(
            name="multi_source_pipeline",
            processors=[
                ExpanderProcessorConfig(
                    type="expander",
                    field_to_expand="items",
                )
            ],
        )

        executor = PipelineExecutor(pipeline_config)

        # Source with multiple records
        async def source() -> AsyncGenerator[BaseRecord, None]:
            yield BaseRecord(
                record_id="rec-1",
                content="first",
                metadata={"items": ["a", "b"]},
            )
            yield BaseRecord(
                record_id="rec-2",
                content="second",
                metadata={"items": ["x", "y", "z"]},
            )

        results = []
        async for result in executor.run(source(), session_id="multi-session"):
            results.append(result)

        # Should produce 2 + 3 = 5 records total
        assert len(results) == 5

        # Verify record IDs
        record_ids = [r.record_id for r in results]
        assert "rec-1_0" in record_ids
        assert "rec-1_1" in record_ids
        assert "rec-2_0" in record_ids
        assert "rec-2_1" in record_ids
        assert "rec-2_2" in record_ids

    @pytest.mark.anyio
    async def test_pipeline_executor_empty_source(self):
        """Verify executor handles empty source gracefully."""
        pipeline_config = PipelineConfig(
            name="empty_pipeline",
            processors=[
                ExpanderProcessorConfig(
                    type="expander",
                    field_to_expand="items",
                )
            ],
        )

        executor = PipelineExecutor(pipeline_config)

        # Empty source
        async def source() -> AsyncGenerator[BaseRecord, None]:
            return
            yield  # Make it a generator

        results = []
        async for result in executor.run(source(), session_id="empty-session"):
            results.append(result)

        # Should produce no results
        assert len(results) == 0


class TestUnifiedProcessorTracing:
    """Test tracing functionality in UnifiedProcessor."""

    @pytest.mark.anyio
    async def test_unified_processor_creates_trace_span(
        self, tracer_provider, get_recorded_spans, clear_recorded_spans
    ):
        """Verify tracing is set up correctly."""
        # Clear any existing spans
        clear_recorded_spans()

        # Create a simple test processor
        class TestProcessor(UnifiedProcessor):
            async def _process_record(
                self, context: ProcessingContext
            ) -> AsyncGenerator[BaseRecord, None]:
                # Just pass through
                yield context.record

        config = ProcessorConfig(type="test", name="test_processor")
        processor = TestProcessor(config)

        record = BaseRecord(record_id="trace-001", content="content")
        context = ProcessingContext(session_id="trace-session", record=record)

        # Process the record
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Verify output
        assert len(outputs) == 1

        # Get recorded spans
        spans = get_recorded_spans()

        # Should have created at least one span
        assert len(spans) > 0

        # Find our processor span
        processor_spans = [s for s in spans if s.name == "processor.test"]
        assert len(processor_spans) == 1

        span = processor_spans[0]

        # Verify span attributes
        attributes = dict(span.attributes)
        assert attributes["processor.name"] == "test_processor"
        assert attributes["processor.type"] == "test"
        assert attributes["record.id"] == "trace-001"

    @pytest.mark.anyio
    async def test_unified_processor_trace_on_error(
        self, tracer_provider, get_recorded_spans, clear_recorded_spans
    ):
        """Verify span records exceptions."""
        clear_recorded_spans()

        # Create processor that raises an error
        class ErrorProcessor(UnifiedProcessor):
            async def _process_record(
                self, context: ProcessingContext
            ) -> AsyncGenerator[BaseRecord, None]:
                raise ValueError("Intentional test error")
                yield  # Make it a generator

        config = ProcessorConfig(type="error_test", name="error_processor")
        processor = ErrorProcessor(config)

        record = BaseRecord(record_id="error-001", content="content")
        context = ProcessingContext(session_id="error-session", record=record)

        # Process should raise
        with pytest.raises(ValueError, match="Intentional test error"):
            async for _ in processor.process(context):
                pass

        # Get spans
        spans = get_recorded_spans()
        assert len(spans) > 0

        # Find error span
        error_spans = [s for s in spans if s.name == "processor.error_test"]
        assert len(error_spans) == 1

        span = error_spans[0]

        # Verify exception was recorded
        events = list(span.events)
        assert len(events) > 0
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) == 1

        # Verify exception details
        exception_event = exception_events[0]
        assert "exception.type" in exception_event.attributes
        assert exception_event.attributes["exception.type"] == "ValueError"
        assert "exception.message" in exception_event.attributes
        assert "Intentional test error" in exception_event.attributes["exception.message"]


class TestProcessorIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.anyio
    async def test_context_metadata_flows_through_pipeline(self):
        """Verify metadata persists across processor chain."""
        # Create pipeline
        pipeline_config = PipelineConfig(
            name="metadata_flow_pipeline",
            processors=[
                ExpanderProcessorConfig(
                    type="expander",
                    field_to_expand="items",
                )
            ],
        )

        executor = PipelineExecutor(pipeline_config)

        # Source with metadata
        async def source() -> AsyncGenerator[BaseRecord, None]:
            yield BaseRecord(
                record_id="meta-001",
                content="content",
                metadata={"items": ["a", "b"], "original": "value"},
            )

        # Create initial context with metadata
        results = []
        async for result in executor.run(source(), session_id="meta-session"):
            results.append(result)

        # Verify metadata persisted
        assert len(results) == 2
        for result in results:
            # Original metadata should be preserved
            assert result.metadata["original"] == "value"
            # Expansion metadata should be added
            assert "expansion_source_id" in result.metadata
            assert result.metadata["expansion_source_id"] == "meta-001"


class TestShellProcessor:
    """Test ShellProcessor for shell command execution."""

    @pytest.mark.anyio
    async def test_shell_processor_executes_command(self):
        """Verify basic command execution."""
        from buttermilk._core.processor_config import ShellProcessorConfig
        from buttermilk.processors.unified_processors import ShellProcessor

        # Create config for a simple echo command
        config = ShellProcessorConfig(
            type="shell",
            command="echo 'Hello, World!'",
            timeout_seconds=5,
        )

        processor = ShellProcessor(config)

        record = BaseRecord(
            record_id="shell-001",
            content="test content",
        )

        context = ProcessingContext(
            session_id="session-shell",
            record=record,
        )

        # Process the record
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield exactly 1 record (the original)
        assert len(outputs) == 1
        assert outputs[0].record_id == "shell-001"

        # Verify stdout is stored in context metadata
        assert "stdout" in context.metadata
        assert context.metadata["stdout"] == "Hello, World!"

    @pytest.mark.anyio
    async def test_shell_processor_stores_stdout_in_metadata(self):
        """Verify output capture in context metadata."""
        from buttermilk._core.processor_config import ShellProcessorConfig
        from buttermilk.processors.unified_processors import ShellProcessor

        config = ShellProcessorConfig(
            type="shell",
            command="echo 'test output'",
        )

        processor = ShellProcessor(config)

        record = BaseRecord(
            record_id="shell-002",
            content="content",
        )

        context = ProcessingContext(
            session_id="session-metadata",
            record=record,
        )

        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Verify metadata contains both stdout and stderr
        assert "stdout" in context.metadata
        assert "stderr" in context.metadata
        assert context.metadata["stdout"] == "test output"
        assert context.metadata["stderr"] == ""

    @pytest.mark.anyio
    async def test_shell_processor_replaces_placeholders(self):
        """Verify {record_id} placeholder replacement."""
        from buttermilk._core.processor_config import ShellProcessorConfig
        from buttermilk.processors.unified_processors import ShellProcessor

        config = ShellProcessorConfig(
            type="shell",
            command="echo 'Processing: {record_id}'",
        )

        processor = ShellProcessor(config)

        record = BaseRecord(
            record_id="test-123",
            content="content",
        )

        context = ProcessingContext(
            session_id="session-placeholder",
            record=record,
        )

        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Verify placeholder was replaced
        assert context.metadata["stdout"] == "Processing: test-123"

    @pytest.mark.anyio
    async def test_shell_processor_raises_on_failure(self):
        """Verify fail-fast on command failure (non-zero exit code)."""
        from buttermilk._core.processor_config import ShellProcessorConfig
        from buttermilk.processors.unified_processors import ShellProcessor

        # Use a command that will fail
        config = ShellProcessorConfig(
            type="shell",
            command="exit 1",
            timeout_seconds=5,
        )

        processor = ShellProcessor(config)

        record = BaseRecord(
            record_id="shell-fail",
            content="content",
        )

        context = ProcessingContext(
            session_id="session-fail",
            record=record,
        )

        # Should raise an error on non-zero exit code
        with pytest.raises(ValueError, match="Shell command failed"):
            async for _ in processor.process(context):
                pass


class TestFilterProcessor:
    """Test FilterProcessor for record filtering based on JMESPath criteria."""

    @pytest.mark.anyio
    async def test_filter_processor_passes_matching_record(self):
        """Verify FilterProcessor yields records that match criteria."""
        from buttermilk._core.processor_config import FilterProcessorConfig
        from buttermilk.processors.unified_processors import FilterProcessor

        # Create config with criteria that will match
        config = FilterProcessorConfig(
            type="filter",
            criteria="metadata.status == 'active'",
        )

        processor = FilterProcessor(config)

        # Create a record that matches the criteria
        record = BaseRecord(
            record_id="filter-001",
            content="test content",
            metadata={"status": "active", "category": "test"},
        )

        context = ProcessingContext(
            session_id="session-filter",
            record=record,
        )

        # Process the record
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield the record since it matches criteria
        assert len(outputs) == 1
        assert outputs[0].record_id == "filter-001"
        assert outputs[0].metadata["status"] == "active"

    @pytest.mark.anyio
    async def test_filter_processor_filters_non_matching_record(self):
        """Verify FilterProcessor yields nothing for non-matching records."""
        from buttermilk._core.processor_config import FilterProcessorConfig
        from buttermilk.processors.unified_processors import FilterProcessor

        # Create config with criteria that won't match
        config = FilterProcessorConfig(
            type="filter",
            criteria="metadata.status == 'active'",
        )

        processor = FilterProcessor(config)

        # Create a record that does NOT match the criteria
        record = BaseRecord(
            record_id="filter-002",
            content="test content",
            metadata={"status": "inactive", "category": "test"},
        )

        context = ProcessingContext(
            session_id="session-filter-no-match",
            record=record,
        )

        # Process the record
        outputs = []
        async for output in processor.process(context):
            outputs.append(output)

        # Should yield nothing since record doesn't match criteria
        assert len(outputs) == 0

    @pytest.mark.anyio
    async def test_filter_processor_uses_jmespath_criteria(self):
        """Verify FilterProcessor evaluates JMESPath expression correctly."""
        from buttermilk._core.processor_config import FilterProcessorConfig
        from buttermilk.processors.unified_processors import FilterProcessor

        # Create config with complex JMESPath criteria
        config = FilterProcessorConfig(
            type="filter",
            criteria="length(metadata.tags) > `2`",
        )

        processor = FilterProcessor(config)

        # Create record with tags list
        record_match = BaseRecord(
            record_id="filter-003",
            content="content",
            metadata={"tags": ["python", "testing", "async"]},  # length = 3
        )

        record_no_match = BaseRecord(
            record_id="filter-004",
            content="content",
            metadata={"tags": ["python"]},  # length = 1
        )

        # Test matching record
        context_match = ProcessingContext(
            session_id="session-jmespath-match",
            record=record_match,
        )

        outputs_match = []
        async for output in processor.process(context_match):
            outputs_match.append(output)

        assert len(outputs_match) == 1
        assert outputs_match[0].record_id == "filter-003"

        # Test non-matching record
        context_no_match = ProcessingContext(
            session_id="session-jmespath-no-match",
            record=record_no_match,
        )

        outputs_no_match = []
        async for output in processor.process(context_no_match):
            outputs_no_match.append(output)

        assert len(outputs_no_match) == 0


class TestProcessorRegistry:
    """Test processor registry for dynamic processor instantiation."""

    def test_registry_creates_processor_from_config(self):
        """Verify registry creates correct processor type from config."""
        from buttermilk._core.processor_registry import create_processor
        from buttermilk.processors.unified_processors import ExpanderProcessor

        # Processors are already registered on module import
        config = ExpanderProcessorConfig(
            type="expander",
            field_to_expand="test_field",
        )

        # Create processor using registry
        processor = create_processor(config)

        # Verify correct type created
        assert isinstance(processor, ExpanderProcessor)
        assert processor.config.field_to_expand == "test_field"

    def test_registry_raises_on_unknown_type(self):
        """Verify KeyError raised for unknown processor types."""
        from buttermilk._core.processor_registry import create_processor

        # Create config with unknown type
        config = ProcessorConfig(type="unknown_processor_type")

        # Should raise KeyError (fail-fast)
        with pytest.raises(KeyError, match="unknown_processor_type"):
            create_processor(config)


class TestBatchProcessor:
    """Test BatchProcessor functionality and integration with executor."""

    @pytest.mark.anyio
    async def test_batch_processor_buffers_records(self):
        """Verify BatchProcessor buffers records until batch_size is reached."""
        # Create a test batch processor that accumulates records
        class TestBatchProcessor(UnifiedBatchProcessor):
            def __init__(self, config: BatchProcessorConfig):
                super().__init__(config)
                self.processed_batches = []

            async def _process_batch(
                self, contexts: list[ProcessingContext]
            ) -> AsyncGenerator[list[BaseRecord], None]:
                # Store batch for verification
                self.processed_batches.append(contexts)
                # Yield records unchanged
                yield [ctx.record for ctx in contexts]

        config = BatchProcessorConfig(
            type="test_batch",
            batch_size=3,
        )

        processor = TestBatchProcessor(config)

        # Create contexts
        contexts = [
            ProcessingContext(
                session_id="batch-session",
                record=BaseRecord(
                    record_id=f"batch-{i}",
                    content=f"content-{i}",
                ),
            )
            for i in range(3)
        ]

        # Process batch
        outputs = []
        async for output_batch in processor.process_batch(contexts):
            outputs.extend(output_batch)

        # Verify all records processed
        assert len(outputs) == 3
        assert len(processor.processed_batches) == 1
        assert len(processor.processed_batches[0]) == 3

        # Verify record IDs
        record_ids = [r.record_id for r in outputs]
        assert record_ids == ["batch-0", "batch-1", "batch-2"]

    @pytest.mark.anyio
    async def test_batch_processor_flushes_on_completion(self):
        """Verify remaining records are processed at end of stream."""
        # Create a test batch processor
        class TestBatchProcessor(UnifiedBatchProcessor):
            def __init__(self, config: BatchProcessorConfig):
                super().__init__(config)
                self.batch_sizes = []

            async def _process_batch(
                self, contexts: list[ProcessingContext]
            ) -> AsyncGenerator[list[BaseRecord], None]:
                # Track batch size
                self.batch_sizes.append(len(contexts))
                # Yield records with metadata indicating batch size
                output_records = []
                for ctx in contexts:
                    record = BaseRecord(
                        record_id=ctx.record.record_id,
                        content=ctx.record.content,
                        metadata={"batch_size": len(contexts)},
                    )
                    output_records.append(record)
                yield output_records

        config = BatchProcessorConfig(
            type="test_batch",
            batch_size=3,
        )

        processor = TestBatchProcessor(config)

        # Create 7 contexts (will need 2 full batches + 1 partial)
        # But for this unit test, we'll manually call process_batch twice
        # to simulate the executor behavior

        # First batch (full)
        batch1_contexts = [
            ProcessingContext(
                session_id="session",
                record=BaseRecord(record_id=f"rec-{i}", content=f"content-{i}"),
            )
            for i in range(3)
        ]

        # Second batch (partial - only 1 record)
        batch2_contexts = [
            ProcessingContext(
                session_id="session",
                record=BaseRecord(record_id="rec-3", content="content-3"),
            )
        ]

        # Process both batches
        all_outputs = []

        async for output_batch in processor.process_batch(batch1_contexts):
            all_outputs.extend(output_batch)

        async for output_batch in processor.process_batch(batch2_contexts):
            all_outputs.extend(output_batch)

        # Verify batch sizes tracked
        assert processor.batch_sizes == [3, 1]

        # Verify all records processed
        assert len(all_outputs) == 4
        assert all_outputs[0].metadata["batch_size"] == 3
        assert all_outputs[3].metadata["batch_size"] == 1

    @pytest.mark.anyio
    async def test_executor_handles_batch_processor(self):
        """Verify PipelineExecutor routes to batch processing correctly."""
        # Create a test batch processor for the pipeline
        class SimpleBatchProcessor(UnifiedBatchProcessor):
            async def _process_batch(
                self, contexts: list[ProcessingContext]
            ) -> AsyncGenerator[list[BaseRecord], None]:
                # Add batch metadata to each record
                output_records = []
                for ctx in contexts:
                    record = BaseRecord(
                        record_id=ctx.record.record_id,
                        content=ctx.record.content,
                        metadata={
                            "batched": True,
                            "batch_size": len(contexts),
                        },
                    )
                    output_records.append(record)
                yield output_records

        # Register the processor temporarily
        from buttermilk._core.processor_registry import register_processor

        register_processor("simple_batch", SimpleBatchProcessor)

        # Create config for batch processor
        batch_config = BatchProcessorConfig(
            type="simple_batch",
            batch_size=2,
        )

        # Create pipeline with batch processor
        pipeline_config = PipelineConfig(
            name="batch_test_pipeline",
            processors=[batch_config],
        )

        executor = PipelineExecutor(pipeline_config)

        # Create source with 3 records
        async def source() -> AsyncGenerator[BaseRecord, None]:
            for i in range(3):
                yield BaseRecord(
                    record_id=f"source-{i}",
                    content=f"content-{i}",
                )

        # Run pipeline
        results = []
        async for result in executor.run(source(), session_id="batch-exec-session"):
            results.append(result)

        # Verify all records processed through batch processor
        assert len(results) == 3

        # Verify batch metadata exists
        for result in results:
            assert "batched" in result.metadata
            assert result.metadata["batched"] is True
            # batch_size should be 2 for first two, 1 for last
            assert result.metadata["batch_size"] in [1, 2]

    @pytest.mark.anyio
    async def test_batch_processor_tracing(
        self, tracer_provider, get_recorded_spans, clear_recorded_spans
    ):
        """Verify batch processor creates appropriate trace spans."""
        clear_recorded_spans()

        # Create a simple batch processor
        class TracedBatchProcessor(UnifiedBatchProcessor):
            async def _process_batch(
                self, contexts: list[ProcessingContext]
            ) -> AsyncGenerator[list[BaseRecord], None]:
                yield [ctx.record for ctx in contexts]

        config = BatchProcessorConfig(
            type="traced_batch",
            name="traced_batch_processor",
            batch_size=2,
        )

        processor = TracedBatchProcessor(config)

        # Create batch
        contexts = [
            ProcessingContext(
                session_id="trace-session",
                record=BaseRecord(record_id=f"trace-{i}", content=f"content-{i}"),
            )
            for i in range(2)
        ]

        # Process batch
        outputs = []
        async for output_batch in processor.process_batch(contexts):
            outputs.extend(output_batch)

        # Verify outputs
        assert len(outputs) == 2

        # Get recorded spans
        spans = get_recorded_spans()
        assert len(spans) > 0

        # Find batch processor span
        batch_spans = [s for s in spans if s.name == "batch_processor.traced_batch"]
        assert len(batch_spans) == 1

        span = batch_spans[0]

        # Verify span attributes
        attributes = dict(span.attributes)
        assert attributes["processor.name"] == "traced_batch_processor"
        assert attributes["processor.type"] == "traced_batch"
        assert attributes["batch.size"] == 2
        assert attributes["batch.output_size"] == 2
