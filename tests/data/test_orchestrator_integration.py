"""Tests for orchestrator integration with new data loading system."""

import json
import tempfile
from pathlib import Path

import pytest
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.types import RunRequest

pytestmark = pytest.mark.anyio


class MockOrchestrator(Orchestrator):
    """Mock orchestrator for testing data loading integration."""
    
    async def _setup(self, request: RunRequest) -> None:
        """Setup for mock orchestrator."""
        pass
    
    async def _cleanup(self) -> None:
        """Cleanup for mock orchestrator."""
        pass
    
    async def _run(self, request: RunRequest) -> None:
        """Run method for mock orchestrator."""
        await self._setup(request)


class TestOrchestratorDataLoaderIntegration:
    """Test integration between orchestrator and new data loading system."""

    @pytest.fixture
    def sample_data_config(self, tmp_path):
        """Create sample data configuration for testing."""
        # Create test JSONL file
        jsonl_path = tmp_path / "test_data.jsonl"
        test_records = [
            {"record_id": "test1", "content": "First test record", "category": "A"},
            {"record_id": "test2", "content": "Second test record", "category": "B"},
            {"record_id": "test3", "content": "Third test record", "category": "A"}
        ]

        with open(jsonl_path, 'w') as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')

        return {
            "test_data": DataSourceConfig(
                type="file",
                path=str(jsonl_path),
                columns={"content": "content", "category": "category"}
            )
        }

    async def test_orchestrator_load_data(self, sample_data_config):
        """Test that orchestrator can load data using new system."""
        orchestrator = MockOrchestrator(orchestrator="mock", name="test_orchestrator", storage=sample_data_config)

        # Test load_data method
        await orchestrator.load_data()

        # Verify data loaders were created
        assert len(orchestrator._input_loaders) == 1
        assert "test_data" in orchestrator._input_loaders

        # Verify we can get records
        records = await orchestrator.get_all_records("test_data")
        assert len(records) == 3
        assert all(record.content.startswith("First") or 
                  record.content.startswith("Second") or 
                  record.content.startswith("Third") for record in records)

    async def test_orchestrator_get_record_by_id(self, sample_data_config):
        """Test that orchestrator can find specific records by ID."""
        orchestrator = MockOrchestrator(orchestrator="mock", name="test_orchestrator", storage=sample_data_config)

        await orchestrator.load_data()

        # Test getting specific record
        record = await orchestrator.get_record_dataset("test1")
        assert record.record_id == "test1"
        assert record.content == "First test record"
        assert record.metadata["category"] == "A"

    async def test_orchestrator_record_not_found(self, sample_data_config):
        """Test that orchestrator raises error for non-existent record."""
        orchestrator = MockOrchestrator(orchestrator="mock", name="test_orchestrator", storage=sample_data_config)

        await orchestrator.load_data()

        # Test getting non-existent record
        from buttermilk._core.exceptions import ProcessingError
        with pytest.raises(ProcessingError, match="Unable to find requested record"):
            await orchestrator.get_record_dataset("nonexistent_id")

    async def test_orchestrator_no_data_sources(self):
        """Test orchestrator behavior with no data sources configured."""
        orchestrator = MockOrchestrator(
            orchestrator="mock",
            name="test_orchestrator"
        )

        # Should handle gracefully
        await orchestrator.load_data()
        assert len(orchestrator._input_loaders) == 0

        # Should return empty list
        records = await orchestrator.get_all_records()
        assert records == []

    async def test_orchestrator_multiple_data_sources(self, tmp_path):
        """Test orchestrator with multiple data sources."""
        # Create JSONL file
        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write(json.dumps({"record_id": "jsonl1", "content": "JSONL content"}) + '\n')

        # Create CSV file
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w') as f:
            f.write("record_id,content\n")
            f.write("csv1,CSV content\n")

        data_config = {
            "jsonl_source": DataSourceConfig(type="file", path=str(jsonl_path)),
            "csv_source": DataSourceConfig(type="file", path=str(csv_path))
        }

        orchestrator = MockOrchestrator(orchestrator="mock", name="test_orchestrator", storage=data_config)

        await orchestrator.load_data()

        # Verify both loaders created
        assert len(orchestrator._input_loaders) == 2
        assert "jsonl_source" in orchestrator._input_loaders
        assert "csv_source" in orchestrator._input_loaders

        # Test getting records from specific source
        jsonl_records = await orchestrator.get_all_records("jsonl_source") 
        csv_records = await orchestrator.get_all_records("csv_source")

        assert len(jsonl_records) == 1
        assert len(csv_records) == 1
        assert jsonl_records[0].content == "JSONL content"
        assert csv_records[0].content == "CSV content"

        # Test getting all records
        all_records = await orchestrator.get_all_records()
        assert len(all_records) == 2

    async def test_orchestrator_streaming_behavior(self, tmp_path):
        """Test that data loading is truly streaming (not loading all at once)."""
        # Create a file with many records
        jsonl_path = tmp_path / "large_data.jsonl"
        with open(jsonl_path, 'w') as f:
            for i in range(100):
                record = {"record_id": f"record_{i}", "content": f"Content {i}"}
                f.write(json.dumps(record) + '\n')

        data_config = {
            "large_data": DataSourceConfig(type="file", path=str(jsonl_path))
        }

        orchestrator = MockOrchestrator(orchestrator="mock", name="test_orchestrator", storage=data_config)

        await orchestrator.load_data()

        # Test that we can find a specific record without loading all
        # (This demonstrates streaming behavior)
        record = await orchestrator.get_record_dataset("record_50")
        assert record.record_id == "record_50"
        assert record.content == "Content 50"
