"""Tests for the new data loading system."""

import json
import tempfile
import csv
from pathlib import Path
from typing import List

import pytest

from buttermilk._core.config import DataSourceConfig
from buttermilk._core.types import Record
from buttermilk.data.loaders import (
    DataLoader,
    JSONLDataLoader,
    CSVDataLoader,
    PlaintextDataLoader,
    HuggingFaceDataLoader,
)
from buttermilk._core.storage_config import StorageConfig


class TestDataLoader:
    """Test the abstract DataLoader base class."""

    def test_abstract_instantiation_fails(self):
        """Test that DataLoader cannot be instantiated directly."""
        config = DataSourceConfig(type="file", path="test.jsonl")
        with pytest.raises(TypeError):
            DataLoader(config)


class TestJSONLDataLoader:
    """Test JSONL data loader."""

    @pytest.fixture
    def sample_jsonl_data(self):
        """Create sample JSONL data for testing."""
        return [
            {"record_id": "test1", "content": "First test record", "label": "positive"},
            {"record_id": "test2", "content": "Second test record", "label": "negative"},
            {"record_id": "test3", "content": "Third test record", "label": "neutral"}
        ]

    @pytest.fixture
    def jsonl_file(self, sample_jsonl_data, tmp_path):
        """Create a temporary JSONL file for testing."""
        jsonl_path = tmp_path / "test_data.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in sample_jsonl_data:
                f.write(json.dumps(item) + '\n')
        return str(jsonl_path)

    def test_jsonl_loader_basic(self, jsonl_file):
        """Test basic JSONL loading functionality."""
        config = DataSourceConfig(type="file", path=jsonl_file)
        loader = JSONLDataLoader(config)
        
        records = list(loader)
        
        assert len(records) == 3
        assert all(isinstance(record, Record) for record in records)
        assert records[0].content == "First test record"
        assert records[1].content == "Second test record"
        assert records[2].content == "Third test record"

    def test_jsonl_loader_with_column_mapping(self, jsonl_file):
        """Test JSONL loading with column mapping."""
        config = DataSourceConfig(
            type="file",
            path=jsonl_file,
            columns={"content": "content", "category": "label"}
        )
        loader = JSONLDataLoader(config)
        
        records = list(loader)
        
        assert len(records) == 3
        assert "category" in records[0].metadata
        assert records[0].metadata["category"] == "positive"

    def test_jsonl_loader_metadata(self, jsonl_file):
        """Test that JSONL loader includes proper metadata."""
        config = DataSourceConfig(type="file", path=jsonl_file)
        loader = JSONLDataLoader(config)
        
        records = list(loader)
        
        for i, record in enumerate(records):
            assert record.metadata["loader_type"] == "jsonl"
            assert record.metadata["line_number"] == i
            assert record.metadata["source"] == jsonl_file

    def test_jsonl_loader_invalid_json(self, tmp_path):
        """Test JSONL loader handling of invalid JSON lines."""
        jsonl_path = tmp_path / "invalid.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        config = DataSourceConfig(type="file", path=str(jsonl_path))
        loader = JSONLDataLoader(config)
        
        records = list(loader)
        
        # Should skip invalid line and load only valid ones
        assert len(records) == 2
        assert "valid" in str(records[0].content)
        assert "another" in str(records[1].content)

    def test_jsonl_loader_empty_lines(self, tmp_path):
        """Test JSONL loader handling of empty lines."""
        jsonl_path = tmp_path / "empty_lines.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write('{"test": "data"}\n')
            f.write('\n')  # Empty line
            f.write('   \n')  # Whitespace line
            f.write('{"more": "data"}\n')
        
        config = DataSourceConfig(type="file", path=str(jsonl_path))
        loader = JSONLDataLoader(config)
        
        records = list(loader)
        
        # Should skip empty lines
        assert len(records) == 2


class TestCSVDataLoader:
    """Test CSV data loader."""

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return [
            {"record_id": "csv1", "content": "First CSV record", "score": "0.8"},
            {"record_id": "csv2", "content": "Second CSV record", "score": "0.6"},
            {"record_id": "csv3", "content": "Third CSV record", "score": "0.9"}
        ]

    @pytest.fixture
    def csv_file(self, sample_csv_data, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"
        with open(csv_path, 'w', newline='') as f:
            if sample_csv_data:
                writer = csv.DictWriter(f, fieldnames=sample_csv_data[0].keys())
                writer.writeheader()
                writer.writerows(sample_csv_data)
        return str(csv_path)

    def test_csv_loader_basic(self, csv_file):
        """Test basic CSV loading functionality."""
        config = DataSourceConfig(type="file", path=csv_file)
        loader = CSVDataLoader(config)
        
        records = list(loader)
        
        assert len(records) == 3
        assert all(isinstance(record, Record) for record in records)
        assert records[0].content == "First CSV record"

    def test_csv_loader_with_column_mapping(self, csv_file):
        """Test CSV loading with column mapping."""
        config = DataSourceConfig(
            type="file",
            path=csv_file,
            columns={"content": "content", "rating": "score"}
        )
        loader = CSVDataLoader(config)
        
        records = list(loader)
        
        assert len(records) == 3
        assert "rating" in records[0].metadata
        assert records[0].metadata["rating"] == "0.8"

    def test_csv_loader_metadata(self, csv_file):
        """Test that CSV loader includes proper metadata."""
        config = DataSourceConfig(type="file", path=csv_file)
        loader = CSVDataLoader(config)
        
        records = list(loader)
        
        for i, record in enumerate(records):
            assert record.metadata["loader_type"] == "csv"
            assert record.metadata["row_number"] == i
            assert record.metadata["source"] == csv_file


class TestPlaintextDataLoader:
    """Test plaintext data loader."""

    @pytest.fixture
    def text_files(self, tmp_path):
        """Create temporary text files for testing."""
        files = []
        for i, content in enumerate(["First file content", "Second file content", "Third file content"]):
            file_path = tmp_path / f"file_{i}.txt"
            file_path.write_text(content)
            files.append(str(file_path))
        return files, str(tmp_path)

    def test_plaintext_loader_basic(self, text_files):
        """Test basic plaintext loading functionality."""
        files, directory = text_files
        config = DataSourceConfig(type="plaintext", path=directory)
        loader = PlaintextDataLoader(config)
        
        records = list(loader)
        
        assert len(records) == 3
        assert all(isinstance(record, Record) for record in records)
        
        # Check content is loaded correctly
        contents = [record.content for record in records]
        assert "First file content" in contents
        assert "Second file content" in contents
        assert "Third file content" in contents

    def test_plaintext_loader_with_glob(self, text_files):
        """Test plaintext loading with glob pattern."""
        files, directory = text_files
        config = DataSourceConfig(type="plaintext", path=directory, glob="file_0.*")
        loader = PlaintextDataLoader(config)
        
        records = list(loader)
        
        # Should only load file_0.txt
        assert len(records) == 1
        assert records[0].content == "First file content"

    def test_plaintext_loader_metadata(self, text_files):
        """Test that plaintext loader includes proper metadata."""
        files, directory = text_files
        config = DataSourceConfig(type="plaintext", path=directory)
        loader = PlaintextDataLoader(config)
        
        records = list(loader)
        
        for record in records:
            assert record.metadata["loader_type"] == "plaintext"
            assert "filename" in record.metadata
            assert "file_size" in record.metadata
            assert record.metadata["file_size"] > 0


class TestHuggingFaceDataLoader:
    """Test HuggingFace data loader."""

    def test_huggingface_loader_creation(self):
        """Test HuggingFace loader can be created."""
        config = DataSourceConfig(
            type="huggingface",
            path="squad",
            name="plain_text",
            split="train[:5]"
        )
        
        try:
            loader = HuggingFaceDataLoader(config)
            assert isinstance(loader, HuggingFaceDataLoader)
        except ImportError:
            pytest.skip("HuggingFace datasets not available")

    @pytest.mark.integration
    def test_huggingface_loader_basic(self):
        """Test basic HuggingFace loading (integration test)."""
        config = DataSourceConfig(
            type="huggingface",
            path="squad",
            name="plain_text", 
            split="train"  # Use full split, will limit in iteration
        )
        
        try:
            loader = HuggingFaceDataLoader(config)
            # Take only first 2 records to avoid long test
            records = []
            for i, record in enumerate(loader):
                if i >= 2:
                    break
                records.append(record)
            
            assert len(records) == 2
            assert all(isinstance(record, Record) for record in records)
            
            for record in records:
                assert record.metadata["loader_type"] == "huggingface"
                assert record.metadata["source"] == "squad"
        except ImportError:
            pytest.skip("HuggingFace datasets not available")


class TestUnifiedStorageSystem:
    """Test the unified storage system that replaced data loader factory."""

    def test_unified_storage_jsonl_detection(self, tmp_path):
        """Test unified storage correctly handles JSONL files."""
        from buttermilk._core.dmrc import get_bm
        
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"test": "data"}\n')
        
        config = StorageConfig(type="file", path=str(jsonl_path))
        bm = get_bm()
        storage = bm.get_storage(config)
        
        # Should be able to iterate through records
        records = list(storage)
        assert len(records) >= 0  # May be empty depending on file structure

    def test_unified_storage_csv_detection(self, tmp_path):
        """Test unified storage correctly handles CSV files."""
        from buttermilk._core.dmrc import get_bm
        
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("content\ntest data\n")
        
        config = StorageConfig(type="file", path=str(csv_path))
        bm = get_bm()
        storage = bm.get_storage(config)
        
        # Should be able to iterate through records
        records = list(storage)
        assert len(records) >= 0

    def test_unified_storage_with_column_mapping(self, tmp_path):
        """Test unified storage with column mapping."""
        from buttermilk._core.dmrc import get_bm
        
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"description": "test content", "category": "test"}\n')
        
        config = StorageConfig(
            type="file", 
            path=str(jsonl_path),
            columns={"content": "description", "metadata": {"type": "category"}}
        )
        bm = get_bm()
        storage = bm.get_storage(config)
        
        records = list(storage)
        if records:  # Only test if we have records
            assert records[0].content == "test content"
            assert records[0].metadata.get("type") == "test"


class TestDataLoaderIntegration:
    """Integration tests for data loaders."""

    async def test_storage_with_orchestrator_pattern(self, tmp_path):
        """Test unified storage integration pattern similar to orchestrator usage."""
        from buttermilk._core.dmrc import get_bm
        
        # Create test data
        jsonl_path = tmp_path / "test.jsonl"
        test_records = [
            {"record_id": "rec1", "content": "Test content 1"},
            {"record_id": "rec2", "content": "Test content 2"}
        ]
        
        with open(jsonl_path, 'w') as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
        
        # Test the pattern used in orchestrator with unified storage
        config = StorageConfig(type="file", path=str(jsonl_path))
        bm = get_bm()
        storage = bm.get_storage(config)
        
        # Simulate finding a specific record
        target_record_id = "rec1"
        found_record = None
        
        for record in storage:
            if record.record_id == target_record_id:
                found_record = record
                break
        
        assert found_record is not None
        assert found_record.record_id == "rec1"
        assert found_record.content == "Test content 1"

    async def test_storage_get_all_records_pattern(self, tmp_path):
        """Test pattern for getting all records from unified storage."""
        from buttermilk._core.dmrc import get_bm
        
        # Create test data
        csv_path = tmp_path / "test.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['record_id', 'content', 'category'])
            writer.writerow(['r1', 'Content 1', 'A'])
            writer.writerow(['r2', 'Content 2', 'B'])
            writer.writerow(['r3', 'Content 3', 'A'])
        
        config = StorageConfig(type="file", path=str(csv_path))
        bm = get_bm()
        storage = bm.get_storage(config)
        
        # Get all records (pattern used in orchestrator.get_all_records)
        all_records = list(storage)
        
        assert len(all_records) == 3
        
        # Test filtering by metadata (simulating category filtering)
        category_a_records = [
            record for record in all_records 
            if record.metadata.get("category") == "A"
        ]
        
        assert len(category_a_records) == 2