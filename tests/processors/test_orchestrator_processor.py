"""Unit tests for OrchestratorProcessor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.orchestrator import OrchestratorProtocol
from buttermilk._core.types import BaseRecord, RunRequest
from buttermilk.processors.orchestrator_processor import OrchestratorProcessor


class TestOrchestratorProcessor:
    """Test suite for OrchestratorProcessor."""

    @pytest.fixture
    def mock_flow_config(self):
        """Create a mock flow configuration."""
        config = MagicMock(spec=OrchestratorProtocol)
        config.name = "test_flow"
        config.description = "Test flow for unit tests"
        config.orchestrator = "buttermilk.orchestrators.mock_orchestrator.MockOrchestrator"
        config.storage = {}
        config.agents = {}
        config.observers = {}
        config.parameters = {}
        return config

    @pytest.fixture
    def sample_record(self):
        """Create a sample record for testing."""
        return BaseRecord(
            record_id="test_record_123",
            metadata={"source": "test"},
        )

    @pytest.mark.anyio
    async def test_skip_cache_default_is_true(self, mock_flow_config):
        """Test that skip_cache defaults to True."""
        # Create processor with default
        processor = OrchestratorProcessor(
            flow_config=mock_flow_config,
            flow_name="test_flow",
        )
        assert processor.skip_cache is True

    @pytest.mark.anyio
    async def test_creates_fresh_orchestrator_per_call(
        self, mock_flow_config, sample_record
    ):
        """Test that each process() call creates a fresh orchestrator instance."""
        orchestrator_instances = []

        def mock_create_orchestrator(flow_config, flow_name):
            """Track orchestrator creation."""
            mock_orch = AsyncMock()
            mock_orch.run = AsyncMock()
            mock_orch.set_bm = MagicMock()
            orchestrator_instances.append(mock_orch)
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
            )

            # Process first record
            results1 = []
            async for result in processor.process(
                sample_record, processor_stage="test_stage"
            ):
                results1.append(result)

            # Process second record
            record2 = BaseRecord(record_id="test_record_456", metadata={})
            results2 = []
            async for result in processor.process(record2, processor_stage="test_stage"):
                results2.append(result)

            # Should have created two separate orchestrator instances
            assert len(orchestrator_instances) == 2
            assert orchestrator_instances[0] is not orchestrator_instances[1]

    @pytest.mark.anyio
    async def test_collects_execution_traces(self, mock_flow_config, sample_record):
        """Test that ExecutionTrace outputs are collected via callback."""
        # Create mock traces that mimic ExecutionTrace behavior
        # We use MagicMock to avoid BM singleton dependency in ExecutionTrace
        mock_trace1 = MagicMock(spec=ExecutionTrace)
        mock_trace1.outputs = {"result": "test_output_1"}

        mock_trace2 = MagicMock(spec=ExecutionTrace)
        mock_trace2.outputs = {"result": "test_output_2"}

        async def mock_run(request: RunRequest):
            """Simulate orchestrator run that calls callback with traces."""
            if request.callback_to_ui:
                await request.callback_to_ui(mock_trace1)
                await request.callback_to_ui(mock_trace2)

        def mock_create_orchestrator(flow_config, flow_name):
            mock_orch = MagicMock()
            mock_orch.run = mock_run
            mock_orch.set_bm = MagicMock()
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
                collect_traces=True,
            )

            results = []
            async for result in processor.process(
                sample_record, processor_stage="test_stage"
            ):
                results.append(result)

            # Should yield one result
            assert len(results) == 1
            result = results[0]

            # Check metadata contains trace info
            assert "test_stage" in result.metadata
            stage_meta = result.metadata["test_stage"]
            assert stage_meta["status"] == "processed"
            assert stage_meta["trace_count"] == 2
            assert len(stage_meta["outputs"]) == 2
            assert stage_meta["outputs"][0] == {"result": "test_output_1"}
            assert stage_meta["outputs"][1] == {"result": "test_output_2"}

    @pytest.mark.anyio
    async def test_passes_record_to_orchestrator(self, mock_flow_config, sample_record):
        """Test that record data is passed to orchestrator in RunRequest."""
        captured_request = None

        async def mock_run(request: RunRequest):
            nonlocal captured_request
            captured_request = request

        def mock_create_orchestrator(flow_config, flow_name):
            mock_orch = MagicMock()
            mock_orch.run = mock_run
            mock_orch.set_bm = MagicMock()
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
            )

            async for _ in processor.process(
                sample_record, processor_stage="test_stage"
            ):
                pass

            # Verify RunRequest was created correctly
            assert captured_request is not None
            assert captured_request.flow == "test_flow"
            assert captured_request.inputs["record_id"] == "test_record_123"
            assert "record" in captured_request.inputs

    @pytest.mark.anyio
    async def test_propagates_orchestrator_errors(self, mock_flow_config, sample_record):
        """Test that orchestrator errors are propagated correctly."""

        async def mock_run(request: RunRequest):
            raise RuntimeError("Orchestrator failed!")

        def mock_create_orchestrator(flow_config, flow_name):
            mock_orch = MagicMock()
            mock_orch.run = mock_run
            mock_orch.set_bm = MagicMock()
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
            )

            with pytest.raises(RuntimeError, match="Orchestrator failed!"):
                async for _ in processor.process(
                    sample_record, processor_stage="test_stage"
                ):
                    pass

    @pytest.mark.anyio
    async def test_collect_traces_disabled(self, mock_flow_config, sample_record):
        """Test that traces are not collected when collect_traces=False."""
        callback_called = False

        async def mock_run(request: RunRequest):
            nonlocal callback_called
            if request.callback_to_ui is not None:
                callback_called = True

        def mock_create_orchestrator(flow_config, flow_name):
            mock_orch = MagicMock()
            mock_orch.run = mock_run
            mock_orch.set_bm = MagicMock()
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
                collect_traces=False,
            )

            results = []
            async for result in processor.process(
                sample_record, processor_stage="test_stage"
            ):
                results.append(result)

            # Should still yield a result
            assert len(results) == 1

            # But callback should not have been called (was None)
            assert not callback_called

    @pytest.mark.anyio
    async def test_preserves_existing_metadata(self, mock_flow_config):
        """Test that existing record metadata is preserved."""
        record = BaseRecord(
            record_id="test_123",
            metadata={"existing_key": "existing_value", "nested": {"key": "value"}},
        )

        async def mock_run(request: RunRequest):
            pass

        def mock_create_orchestrator(flow_config, flow_name):
            mock_orch = MagicMock()
            mock_orch.run = mock_run
            mock_orch.set_bm = MagicMock()
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
            )

            results = []
            async for result in processor.process(record, processor_stage="test_stage"):
                results.append(result)

            result = results[0]

            # Existing metadata should be preserved
            assert result.metadata["existing_key"] == "existing_value"
            assert result.metadata["nested"]["key"] == "value"

            # New stage metadata should be added
            assert "test_stage" in result.metadata
            assert result.metadata["test_stage"]["status"] == "processed"

    @pytest.mark.anyio
    async def test_finalize_processing_returns_true(self, mock_flow_config):
        """Test that finalize_processing always returns True."""
        processor = OrchestratorProcessor(
            flow_config=mock_flow_config,
            flow_name="test_flow",
        )

        result = await processor.finalize_processing()
        assert result is True

    @pytest.mark.anyio
    async def test_injects_bm_into_orchestrator(self, mock_flow_config, sample_record):
        """Test that session-scoped BM is injected into orchestrator."""
        mock_bm = MagicMock()
        set_bm_called = False

        async def mock_run(request: RunRequest):
            pass

        def mock_create_orchestrator(flow_config, flow_name):
            mock_orch = MagicMock()
            mock_orch.run = mock_run

            def track_set_bm(bm):
                nonlocal set_bm_called
                set_bm_called = True
                assert bm is mock_bm

            mock_orch.set_bm = track_set_bm
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
                bm=mock_bm,
            )

            async for _ in processor.process(
                sample_record, processor_stage="test_stage"
            ):
                pass

            assert set_bm_called

    @pytest.mark.anyio
    async def test_passes_parameters_to_run_request(
        self, mock_flow_config, sample_record
    ):
        """Test that OrchestratorProcessor passes parameters field to RunRequest.

        Acceptance criterion: OrchestratorProcessor should accept a parameters
        field and pass it to the RunRequest when process() is called.

        Expected failure: ValidationError or AttributeError because parameters
        field doesn't exist on OrchestratorProcessor yet.
        """
        captured_request = None

        async def mock_run(request: RunRequest):
            nonlocal captured_request
            captured_request = request

        def mock_create_orchestrator(flow_config, flow_name):
            mock_orch = MagicMock()
            mock_orch.run = mock_run
            mock_orch.set_bm = MagicMock()
            return mock_orch

        with patch(
            "buttermilk.processors.orchestrator_processor.OrchestratorFactory.create_orchestrator",
            side_effect=mock_create_orchestrator,
        ):
            # Create processor with parameters field
            test_parameters = {"max_retries": 3, "timeout": 30}
            processor = OrchestratorProcessor(
                flow_config=mock_flow_config,
                flow_name="test_flow",
                parameters=test_parameters,
            )

            async for _ in processor.process(
                sample_record, processor_stage="test_stage"
            ):
                pass

            # Verify RunRequest received the parameters
            assert captured_request is not None
            assert hasattr(captured_request, "parameters")
            assert captured_request.parameters == test_parameters
