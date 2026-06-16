"""Tests for the multi-step `history` accumulator (Proposal v4, task buttermilk-d33698f7).

Covers the data-shape contract and the topology/fail-loud behaviours that the strategic
reviews required be locked down by executable tests:

- StepResult <-> ExecutionTrace single-projection correspondence (no field-name drift).
- Fail-loud `_resolve_inputs`: required input missing as None OR empty list `[]` raises;
  `optional_inputs` are dropped silently.
- The API semaphore actually bounds concurrency under `asyncio.gather` (the mechanism
  FanInLLMProcessor relies on for batch-safe panels).
- FanInLLMProcessor: appends one StepResult per member with stable indices, retains failed
  members as error entries, and enforces a fail-loud completeness gate.

Mocks are confined to the external LLM boundary (the chat client). Internal buttermilk code
runs for real.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.context import ApiSemaphoreContext, set_api_semaphore
from buttermilk._core.contract import ExecutionTrace, StepResult
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import CreateResult
from buttermilk._core.messages import RequestUsage
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.processors.unified_processors import FanInLLMProcessor, LLMProcessor


def _ok_result(content: str = "ok") -> CreateResult:
    return CreateResult(
        content=content,
        finish_reason="stop",
        usage=RequestUsage(prompt_tokens=5, completion_tokens=5),
        cached=False,
    )


class TestStepResultCorrespondence:
    """StepResult must mirror ExecutionTrace's load-bearing fields via one projection."""

    def test_from_execution_trace_preserves_shared_fields(self):
        trace = ExecutionTrace(
            agent_info={"component_name": "judge"},
            outputs={"reasons": ["because"]},
            metadata={"llm_config": {"model": "m1", "template": "judge"}, "template": {"template_hash": "abc"}},
        )
        sr = StepResult.from_execution_trace(trace, step="judge", index=3)

        # Shared/load-bearing fields line up with the trace
        assert sr.outputs == trace.outputs
        assert sr.error == trace.error  # None on success
        assert sr.agent_id == "judge#3"
        assert sr.step == "judge"
        assert sr.index == 3
        # Lean provenance is projected; the bulky record metadata is NOT copied
        assert sr.metadata == {"trace_id": trace.call_id, "model": "m1", "template": "judge", "template_hash": "abc"}

    def test_error_trace_projects_to_error_entry(self):
        trace = ExecutionTrace(
            agent_info={"component_name": "judge"},
            error={"event": "boom", "details": {"error_type": "ValueError"}},
        )
        sr = StepResult.from_execution_trace(trace, step="judge", index=1)
        assert sr.outputs is None
        assert sr.error == {"event": "boom", "details": {"error_type": "ValueError"}}

    def test_content_is_computed_not_stored(self):
        """`content` is a convenience property and must never appear in the serialized entry."""
        sr = StepResult(step="judge", agent_id="judge#0", outputs={"x": 1})
        assert sr.content == str({"x": 1})
        dumped = sr.model_dump()
        assert "content" not in dumped
        # `error` is always present (null on success) so the `error==null` filter is predictable
        assert "error" in dumped and dumped["error"] is None


class TestFailLoudResolveInputs:
    """Required inputs that resolve to None OR empty list `[]` must raise (research integrity)."""

    def _ctx(self, metadata):
        record = BaseRecord(record_id="r1", content="c", metadata=metadata)
        return ProcessingContext(session_id="s", record=record)

    def test_missing_required_input_none_raises(self):
        proc = LLMProcessor(model="m", template="t", inputs={"draft": "record.metadata.absent"})
        with pytest.raises(ProcessingError, match="Required input 'draft'"):
            proc._resolve_inputs(self._ctx({}))

    def test_missing_required_input_empty_list_raises(self):
        """A no-match JMESPath filter returns `[]`, not None — must still fail loud."""
        proc = LLMProcessor(model="m", template="t", inputs={"answers": "record.metadata.history[?step=='judge']"})
        with pytest.raises(ProcessingError, match="empty list"):
            proc._resolve_inputs(self._ctx({"history": [{"step": "critic", "error": None}]}))

    def test_optional_input_dropped_silently(self):
        proc = LLMProcessor(model="m", template="t", inputs={"extra": "record.metadata.absent"}, optional_inputs=["extra"])
        assert proc._resolve_inputs(self._ctx({})) == {}

    def test_config_override_inputs_are_optional_by_default(self):
        """`model`/`template` overrides fall back to defaults when absent — never fail loud."""
        proc = LLMProcessor(model="m", template="t", inputs={"model": "record.metadata.absent"})
        assert proc._resolve_inputs(self._ctx({})) == {}

    def test_present_required_input_resolves(self):
        proc = LLMProcessor(model="m", template="t", inputs={"answers": "record.metadata.history[?step=='judge' && error==null]"})
        resolved = proc._resolve_inputs(self._ctx({"history": [{"step": "judge", "error": None, "outputs": {"r": 1}}]}))
        assert resolved["answers"] == [{"step": "judge", "error": None, "outputs": {"r": 1}}]


class TestApiSemaphoreBoundsConcurrency:
    """FanInLLMProcessor relies on the global API semaphore to cap concurrent LLM calls."""

    @pytest.mark.anyio
    async def test_gather_concurrency_is_bounded(self):
        set_api_semaphore(asyncio.Semaphore(2))
        try:
            active = 0
            peak = 0

            async def worker():
                nonlocal active, peak
                async with ApiSemaphoreContext():
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(0.01)
                    active -= 1

            await asyncio.gather(*[worker() for _ in range(10)])
            assert peak == 2
        finally:
            set_api_semaphore(asyncio.Semaphore(1000))


class TestFanInLLMProcessor:
    """Independent panel: one StepResult per member, stable indices, fail-loud completeness."""

    def _ctx(self):
        # `test/simple` requires a `var` template parameter (flattened from record metadata).
        record = BaseRecord(record_id="panel-1", content="content", metadata={"var": "test"})
        return ProcessingContext(session_id="s", record=record)

    @pytest.mark.anyio
    async def test_appends_one_step_result_per_member(self):
        proc = FanInLLMProcessor(
            name="judge",
            model="gpt-4",
            template="test/simple",
            members=[{"model": "gpt-4"}, {"model": "claude-3-opus"}, {"model": "gemini"}],
        )
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.return_value = _ok_result("verdict")
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = [o async for o in proc.process(self._ctx())]

        assert len(outputs) == 1
        history = outputs[0].metadata["history"]
        assert len(history) == 3
        assert all(h["step"] == "judge" for h in history)
        assert sorted(h["index"] for h in history) == [0, 1, 2]
        assert {h["agent_id"] for h in history} == {"judge#0", "judge#1", "judge#2"}
        assert all(h["error"] is None for h in history)

    @pytest.mark.anyio
    async def test_failed_member_retained_and_gate_relaxed(self):
        """A failed member is retained as an error entry; min_results lets the panel pass."""
        proc = FanInLLMProcessor(
            name="judge",
            model="gpt-4",
            template="test/simple",
            members=[{}, {}, {}],
            min_results=2,
        )
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        # Exactly one of the three calls fails (order-independent under gather).
        mock_client.call_chat.side_effect = [_ok_result(), RuntimeError("model unavailable"), _ok_result()]
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            outputs = [o async for o in proc.process(self._ctx())]

        history = outputs[0].metadata["history"]
        assert len(history) == 3  # failed member retained, not dropped
        assert sum(1 for h in history if h["error"] is None) == 2
        assert sum(1 for h in history if h["error"] is not None) == 1
        assert sorted(h["index"] for h in history) == [0, 1, 2]  # indices never reindexed on failure

    @pytest.mark.anyio
    async def test_completeness_gate_fails_loud(self):
        """Default gate requires ALL members to succeed → a shortfall raises ProcessingError."""
        proc = FanInLLMProcessor(
            name="judge",
            model="gpt-4",
            template="test/simple",
            members=[{}, {}, {}],
        )
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.side_effect = RuntimeError("all down")
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            with pytest.raises(ProcessingError, match="completeness gate failed"):
                _ = [o async for o in proc.process(self._ctx())]
