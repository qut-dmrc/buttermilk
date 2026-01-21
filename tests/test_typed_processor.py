"""Tests for Typed Data Flow Architecture.

TDD RED phase: These tests define the new processor contract where
processors yield arbitrary BaseModel types and context handles observability.

Contract:
    async def process(input: T, ctx: ProcessingContext) -> AsyncGenerator[U, None]
"""

from typing import AsyncGenerator

import pytest
from pydantic import BaseModel, Field

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord

# --- Test Output Types ---


class JudgeOutput(BaseModel):
    """Example typed output from a judge processor."""

    prediction: bool
    confidence: str  # "high", "medium", "low"
    reasoning: str
    reasons: list[str] = Field(default_factory=list)


class SummaryOutput(BaseModel):
    """Example typed output from a summarizer."""

    summary: str
    key_points: list[str] = Field(default_factory=list)


# --- Context I/O Capture Tests ---


class TestProcessingContextIOCapture:
    """Test that ProcessingContext captures inputs and outputs for tracing."""

    def test_context_has_input_capture_method(self):
        """ProcessingContext should have capture_input method."""
        record = BaseRecord(record_id="test-001", content="test")
        ctx = ProcessingContext(session_id="session-test", record=record)

        # Should have capture_input method
        assert hasattr(ctx, "capture_input"), "ProcessingContext needs capture_input method"

    def test_context_has_output_capture_method(self):
        """ProcessingContext should have capture_output method."""
        record = BaseRecord(record_id="test-001", content="test")
        ctx = ProcessingContext(session_id="session-test", record=record)

        # Should have capture_output method
        assert hasattr(ctx, "capture_output"), "ProcessingContext needs capture_output method"

    def test_context_captures_input(self):
        """capture_input stores the input for tracing."""
        record = BaseRecord(record_id="test-001", content="test content")
        ctx = ProcessingContext(session_id="session-test", record=record)

        # Capture input
        ctx.capture_input(record)

        # Should be stored
        assert ctx.input is not None, "Input should be captured"
        assert ctx.input.record_id == "test-001"

    def test_context_captures_multiple_outputs(self):
        """capture_output accumulates outputs for 1:N processors."""
        record = BaseRecord(record_id="test-001", content="test")
        ctx = ProcessingContext(session_id="session-test", record=record)

        # Capture multiple outputs
        output1 = JudgeOutput(prediction=True, confidence="high", reasoning="test1")
        output2 = JudgeOutput(prediction=False, confidence="low", reasoning="test2")

        ctx.capture_output(output1)
        ctx.capture_output(output2)

        # Should have both
        assert hasattr(ctx, "outputs"), "ProcessingContext needs outputs list"
        assert len(ctx.outputs) == 2
        assert ctx.outputs[0].prediction is True
        assert ctx.outputs[1].prediction is False


# --- Typed Processor Contract Tests ---


class TestTypedProcessorYieldsArbitraryTypes:
    """Test that processors can yield any BaseModel, not just BaseRecord."""

    @pytest.mark.anyio
    async def test_processor_can_yield_non_baserecord_type(self):
        """Processor should be able to yield JudgeOutput directly."""
        from buttermilk._core.processor_core import ProcessorCore

        class MockJudgeProcessor(ProcessorCore):
            """Test processor that yields JudgeOutput."""

            async def _process_record(self, context: ProcessingContext) -> AsyncGenerator[JudgeOutput, None]:
                # Yield typed output, not BaseRecord
                yield JudgeOutput(
                    prediction=True,
                    confidence="high",
                    reasoning="Test reasoning",
                    reasons=["reason1", "reason2"],
                )

        processor = MockJudgeProcessor()
        record = BaseRecord(record_id="test-001", content="test content")
        ctx = ProcessingContext(session_id="session-test", record=record)

        outputs = []
        async for output in processor.process(ctx):
            outputs.append(output)

        # Should yield JudgeOutput, not BaseRecord
        assert len(outputs) == 1
        assert isinstance(outputs[0], JudgeOutput), f"Expected JudgeOutput, got {type(outputs[0])}"
        assert outputs[0].prediction is True
        assert outputs[0].confidence == "high"

    @pytest.mark.anyio
    async def test_processor_can_yield_different_output_types(self):
        """Different processors can yield different output types."""
        from buttermilk._core.processor_core import ProcessorCore

        class SummarizerProcessor(ProcessorCore):
            async def _process_record(self, context: ProcessingContext) -> AsyncGenerator[SummaryOutput, None]:
                yield SummaryOutput(
                    summary="This is a summary",
                    key_points=["point1", "point2"],
                )

        processor = SummarizerProcessor()
        record = BaseRecord(record_id="test-001", content="test content")
        ctx = ProcessingContext(session_id="session-test", record=record)

        outputs = []
        async for output in processor.process(ctx):
            outputs.append(output)

        assert len(outputs) == 1
        assert isinstance(outputs[0], SummaryOutput)
        assert outputs[0].summary == "This is a summary"


# --- Pipeline Typed Flow Tests ---


class TestPipelineTypedFlow:
    """Test that pipeline executor passes typed objects between stages."""

    @pytest.mark.anyio
    async def test_pipeline_passes_typed_output_to_next_stage(self):
        """Output from stage 1 should be input to stage 2."""
        from buttermilk._core.executor import PipelineExecutor
        from buttermilk._core.pipeline_config import PipelineConfig
        from buttermilk._core.processor_core import ProcessorCore

        # Stage 1: BaseRecord -> JudgeOutput
        class Stage1(ProcessorCore):
            async def _process_record(self, ctx: ProcessingContext):
                yield JudgeOutput(
                    prediction=True,
                    confidence="high",
                    reasoning="Stage 1 output",
                )

        # Stage 2: JudgeOutput -> SummaryOutput
        # This should receive JudgeOutput as input
        class Stage2(ProcessorCore):
            async def _process_record(self, ctx: ProcessingContext):
                # The input should be JudgeOutput from Stage 1
                input_obj = ctx.record  # or however we access input
                assert isinstance(input_obj, (JudgeOutput, BaseRecord)), f"Expected typed input, got {type(input_obj)}"

                yield SummaryOutput(
                    summary=f"Summarized: {getattr(input_obj, 'reasoning', 'N/A')}",
                    key_points=["processed"],
                )

        pipeline_config = PipelineConfig(
            name="typed_flow_test",
            processors=[Stage1(), Stage2()],
        )
        executor = PipelineExecutor(pipeline_config)

        async def source():
            yield BaseRecord(record_id="test-001", content="test")

        results = []
        async for result in executor.run(source(), session_id="test-session"):
            results.append(result)

        # Final output should be SummaryOutput
        assert len(results) == 1
        assert isinstance(results[0], SummaryOutput)


class TestLLMProcessorTypedOutput:
    """Test LLMProcessor typed output behavior."""

    def test_llm_processor_backwards_compatibility_removed(self):
        """Verify that output_col and yield_typed (legacy fields) are removed."""
        from buttermilk.processors.unified_processors import LLMProcessor

        processor = LLMProcessor(
            model="test-model",
            template="test_template",
        )

        # Confirm fields are GONE
        assert not hasattr(processor, "output_col")
        assert not hasattr(processor, "yield_typed")

        # Confirm inner core doesn't have them either
        assert not hasattr(processor._llm_core, "output_col")

    def test_llm_processor_has_output_model_field(self):
        """LLMProcessor should have output_model field for typed output."""
        from buttermilk.processors.unified_processors import LLMProcessor

        processor = LLMProcessor(
            model="test-model",
            template="test_template",
            output_model="buttermilk.agents.judge.JudgeReasons",
        )
        assert processor.output_model == "buttermilk.agents.judge.JudgeReasons"
