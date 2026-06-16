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
from buttermilk.utils.templating import render_template


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

    @pytest.mark.anyio
    async def test_default_gate_fails_on_partial_panel(self):
        """The default (all-required) gate must FAIL a 2/3-success panel — the boundary case."""
        proc = FanInLLMProcessor(name="judge", model="gpt-4", template="test/simple", members=[{}, {}, {}])
        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.side_effect = [_ok_result(), RuntimeError("one down"), _ok_result()]
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            with pytest.raises(ProcessingError, match="2/3 members succeeded, required 3"):
                _ = [o async for o in proc.process(self._ctx())]


class TestStructuralCorrespondence:
    """W1: guard against StepResult <-> ExecutionTrace drift by field NAME, not just value.

    A value-snapshot test passes even if `from_execution_trace` silently stops copying a
    renamed/removed field. These assertions fail loudly in that case.
    """

    def test_projection_source_fields_exist_on_execution_trace(self):
        # If any of these is renamed/removed on ExecutionTrace, from_execution_trace would
        # silently drop it — this must break instead.
        for field in ("outputs", "error", "metadata", "call_id"):
            assert field in ExecutionTrace.model_fields, f"ExecutionTrace lost '{field}' — projection would silently break"
        for field in ("step", "index", "agent_id", "outputs", "error", "metadata"):
            assert field in StepResult.model_fields, f"StepResult lost '{field}'"

    def test_projection_copies_each_shared_field(self):
        """Distinct sentinel on every shared field; assert each is carried by the projection."""
        trace = ExecutionTrace(
            agent_info={"component_name": "judge"},
            outputs={"k": "v"},
            error=None,
            metadata={"llm_config": {"model": "M", "template": "T"}, "template": {"template_hash": "H"}},
        )
        sr = StepResult.from_execution_trace(trace, step="judge", index=0)
        assert sr.outputs is trace.outputs  # copied, not reconstructed
        assert sr.error == trace.error
        assert sr.metadata["trace_id"] == trace.call_id
        assert (sr.metadata["model"], sr.metadata["template"], sr.metadata["template_hash"]) == ("M", "T", "H")


class TestNestedPanelToSynthesis:
    """W2 + N1: the E3-shaped chain (panel -> synthesis) the whole design exists to enable.

    Exercises step-first selection with an error-aware filter over a MIXED success/error
    history end-to-end, and confirms chaining a downstream step does not disturb prior
    selection. Also documents the two coexisting agent_id conventions in one history list.
    """

    @pytest.mark.anyio
    async def test_panel_then_synthesis_selects_only_successful_judges(self):
        # 1) Run a judge panel where exactly one member fails (retained as an error entry).
        panel = FanInLLMProcessor(name="judge", model="gpt-4", template="test/simple", members=[{}, {}, {}], min_results=2)
        record = BaseRecord(record_id="e3-1", content="content", metadata={"var": "test"})
        ctx = ProcessingContext(session_id="s", record=record)

        mock_bm = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_chat.side_effect = [_ok_result(), RuntimeError("judge down"), _ok_result()]
        mock_bm.llms.get_client.return_value = mock_client
        mock_bm.llms.connections = {}

        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            panel_out = [o async for o in panel.process(ctx)][0]

        # 2) A downstream synthesis step selects ONLY the successful judges (error-aware filter).
        synth = LLMProcessor(
            name="synthesize",
            model="gpt-4",
            template="test/simple",
            fail_on_unfilled_parameters=False,
            inputs={"answers": "record.metadata.history[?step=='judge' && error==null]"},
        )
        ctx2 = ProcessingContext(session_id="s", record=panel_out)

        # The resolved input feeding the synthesis step is exactly the 2 successful judges.
        resolved = synth._resolve_inputs(ctx2)
        assert len(resolved["answers"]) == 2
        assert all(a["error"] is None and a["step"] == "judge" for a in resolved["answers"])

        mock_client.call_chat.side_effect = None
        mock_client.call_chat.return_value = _ok_result("final")
        with patch("buttermilk._core.llm_core.bm", mock_bm), patch("buttermilk.processors.unified_processors.bm", mock_bm):
            synth_out = [o async for o in synth.process(ctx2)][0]

        # 3) Chaining preserves prior entries; step-first selection still isolates the judges.
        history = synth_out.metadata["history"]
        assert len(history) == 4  # 3 judges (incl. the retained error) + 1 synthesize
        judges = [h for h in history if h["step"] == "judge"]
        synths = [h for h in history if h["step"] == "synthesize"]
        assert len(judges) == 3 and sum(1 for j in judges if j["error"] is None) == 2
        assert len(synths) == 1 and synths[0]["error"] is None
        # error-aware selection over the full chained history still yields only the 2 successes
        import jmespath

        ok_judges = jmespath.search("history[?step=='judge' && error==null]", {"history": history})
        assert len(ok_judges) == 2


class TestTemplatesRenderHistory:
    """W3: the synthesise/differences templates must address typed StepResult fields for real.

    Renders the actual template files against a history-derived answers list, proving the
    `answer.agent_id` / `answer.outputs` contract executes (not only implicitly in prod).
    """

    @pytest.mark.parametrize("template", ["synthesise", "differences"])
    def test_template_addresses_typed_fields(self, template):
        answers = [
            StepResult(step="judge", index=0, agent_id="judge#0", outputs={"reasons": ["clear breach"]}).model_dump(),
            StepResult(step="judge", index=1, agent_id="judge#1", outputs={"reasons": ["no breach"]}).model_dump(),
        ]
        rendered = render_template(template, template_vars={"answers": answers}, fail_on_unfilled=False).rendered
        # agent_id wrappers and the typed outputs both appear → typed-field addressing works
        assert "judge#0" in rendered and "judge#1" in rendered
        assert "clear breach" in rendered and "no breach" in rendered
