"""Tests for ToxicityModel.process() Processor protocol implementation."""

import pytest

torch = pytest.importorskip("torch")

from buttermilk._core.types import Record
from buttermilk.toxicity.types import EvalRecord, Score


class TestToxicityModelProcess:
    """Test ToxicityModel implements Processor protocol via process() method."""

    @pytest.fixture
    def mock_toxicity_model(self, mocker):
        """Create a mock ToxicityModel with controlled moderate() output."""
        from buttermilk.toxicity.toxicity import ToxicityModel

        # Create a concrete subclass since ToxicityModel has abstract methods
        class MockToxicityModel(ToxicityModel):
            model: str = "mock_model"
            process_chain: str = "mock"
            standard: str = "mock_standard"

            def init_client(self):
                self.client = "mock_client"

            def make_prompt(self, content):
                return content

            def interpret(self, response):
                return EvalRecord(
                    prediction=True,
                    scores=[Score(measure="toxicity", score=0.9)],
                    labels=["toxic"],
                )

            def call_client(self, prompt, **kwargs):
                return {"mock": "response"}

        return MockToxicityModel()

    @pytest.fixture
    def test_record(self) -> Record:
        """Sample record for testing."""
        return Record(
            record_id="test_123",
            content="This is test content for toxicity analysis",
            metadata={"source": "test"},
        )

    @pytest.mark.anyio
<<<<<<< HEAD
    async def test_process_yields_enriched_record(self, mock_toxicity_model, test_record):
        """process() should yield record with toxicity results in metadata."""
        results = []
        async for result in mock_toxicity_model.process(test_record, processor_stage="toxicity_check"):
=======
    async def test_process_yields_enriched_record(
        self, mock_toxicity_model, test_record
    ):
        """process() should yield record with toxicity results in metadata."""
        results = []
        async for result in mock_toxicity_model.process(
            test_record, processor_stage="toxicity_check"
        ):
>>>>>>> origin/stable
            results.append(result)

        assert len(results) == 1
        result = results[0]

        # Original record_id preserved
        assert result.record_id == "test_123"

        # Toxicity results stored in metadata
        assert "toxicity_check" in result.metadata
        tox_result = result.metadata["toxicity_check"]
        assert tox_result["prediction"] is True
        assert tox_result["model"] == "mock_model"
        assert tox_result["standard"] == "mock_standard"
        assert len(tox_result["scores"]) == 1
        assert tox_result["labels"] == ["toxic"]

    @pytest.mark.anyio
<<<<<<< HEAD
    async def test_process_preserves_existing_metadata(self, mock_toxicity_model, test_record):
        """process() should preserve existing metadata."""
        results = []
        async for result in mock_toxicity_model.process(test_record, processor_stage="tox"):
=======
    async def test_process_preserves_existing_metadata(
        self, mock_toxicity_model, test_record
    ):
        """process() should preserve existing metadata."""
        results = []
        async for result in mock_toxicity_model.process(
            test_record, processor_stage="tox"
        ):
>>>>>>> origin/stable
            results.append(result)

        result = results[0]
        # Original metadata preserved
        assert result.metadata["source"] == "test"
        # New toxicity metadata added
        assert "tox" in result.metadata

    @pytest.mark.anyio
    async def test_process_raises_on_empty_content(self, mock_toxicity_model):
        """process() should raise ValueError if record has no content."""
        # BaseRecord allows None content, but Record validates it
        # Create a BaseRecord-like object with no content
        from buttermilk._core.types import BaseRecord

        empty_record = BaseRecord(record_id="empty_test", content=None)

        with pytest.raises(ValueError, match="no content"):
<<<<<<< HEAD
            async for _ in mock_toxicity_model.process(empty_record, processor_stage="tox"):
=======
            async for _ in mock_toxicity_model.process(
                empty_record, processor_stage="tox"
            ):
>>>>>>> origin/stable
                pass

    @pytest.mark.anyio
    async def test_process_handles_error_in_eval_record(self, mocker):
        """process() should include error field from EvalRecord."""
        from buttermilk.toxicity.toxicity import ToxicityModel

        class ErrorToxicityModel(ToxicityModel):
            model: str = "error_model"
            process_chain: str = "test"
            standard: str = "test"

            def init_client(self):
                self.client = "mock"

            def make_prompt(self, content):
                return content

            def interpret(self, response):
                return EvalRecord(
                    prediction=None,
                    error="Failed to interpret response",
                )

            def call_client(self, prompt, **kwargs):
                return {}

        model = ErrorToxicityModel()
        record = Record(record_id="test", content="test content")

        results = []
        async for result in model.process(record, processor_stage="tox"):
            results.append(result)

        assert results[0].metadata["tox"]["error"] == "Failed to interpret response"
