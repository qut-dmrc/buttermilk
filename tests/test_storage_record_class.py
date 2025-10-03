"""Tests for storage record class instantiation."""

import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.storage_config import BaseStorageConfig, BigQueryStorageConfig, FileStorageConfig
from buttermilk._core.types import BaseRecord, Record
from buttermilk.storage.base import Storage
from buttermilk.storage.file import FileStorage
from buttermilk.tools.catalog_test import Title


class TestStorageRecordClass:
    """Test record class resolution and instantiation in storage."""

    def test_base_storage_config_accepts_record_class(self):
        """Test that BaseStorageConfig accepts record_class field."""
        config = BaseStorageConfig(
            type="test",
            dataset_name="test_dataset",
            record_class="buttermilk.tools.catalog_test.Title"
        )

        assert config.record_class == "buttermilk.tools.catalog_test.Title"

    def test_storage_default_record_class(self):
        """Test that storage defaults to Record when no class specified."""
        config = BaseStorageConfig(type="test", dataset_name="test")

        # Create a concrete Storage subclass for testing
        class TestStorage(Storage):
            def __iter__(self):
                return iter([])
            def save(self, records):
                pass
            def count(self):
                return 0

        storage = TestStorage(config)

        # Get the record class
        record_class = storage._get_record_class()
        assert record_class is Record

    def test_storage_custom_record_class(self):
        """Test that storage uses custom record class when specified."""
        config = BaseStorageConfig(
            type="test",
            dataset_name="test",
            record_class="buttermilk.tools.catalog_test.Title"
        )

        class TestStorage(Storage):
            def __iter__(self):
                return iter([])
            def save(self, records):
                pass
            def count(self):
                return 0

        storage = TestStorage(config)

        # Get the record class
        record_class = storage._get_record_class()
        assert record_class is Title

    def test_storage_invalid_record_class_falls_back(self):
        """Test that invalid record class falls back to Record."""
        config = BaseStorageConfig(
            type="test",
            dataset_name="test",
            record_class="nonexistent.module.Class"
        )

        class TestStorage(Storage):
            def __iter__(self):
                return iter([])
            def save(self, records):
                pass
            def count(self):
                return 0

        storage = TestStorage(config)

        # Should fall back to Record with a warning
        with patch('buttermilk.storage.base.logger') as mock_logger:
            record_class = storage._get_record_class()
            assert record_class is Record
            mock_logger.warning.assert_called()

    def test_storage_non_baserecord_class_falls_back(self):
        """Test that non-BaseRecord class falls back to Record."""
        config = BaseStorageConfig(
            type="test",
            dataset_name="test",
            record_class="datetime.datetime"  # Valid class but not BaseRecord
        )

        class TestStorage(Storage):
            def __iter__(self):
                return iter([])
            def save(self, records):
                pass
            def count(self):
                return 0

        storage = TestStorage(config)

        # Should fall back to Record with a warning
        with patch('buttermilk.storage.base.logger') as mock_logger:
            record_class = storage._get_record_class()
            assert record_class is Record
            mock_logger.warning.assert_called()

    def test_create_record_with_default_class(self):
        """Test _create_record() with default Record class."""
        config = BaseStorageConfig(type="test", dataset_name="test")

        class TestStorage(Storage):
            def __iter__(self):
                return iter([])
            def save(self, records):
                pass
            def count(self):
                return 0

        storage = TestStorage(config)

        # Create a record
        record = storage._create_record(
            record_id="123",
            content="test content",
            dataset_name="test"
        )

        assert isinstance(record, Record)
        assert record.record_id == "123"
        assert record.content == "test content"

    def test_create_record_with_title_class(self):
        """Test _create_record() with Title class."""
        config = BaseStorageConfig(
            type="test",
            dataset_name="test",
            record_class="buttermilk.tools.catalog_test.Title"
        )

        class TestStorage(Storage):
            def __iter__(self):
                return iter([])
            def save(self, records):
                pass
            def count(self):
                return 0

        storage = TestStorage(config)

        # Create a Title record
        record = storage._create_record(
            record_id="tmdb_123",
            title="Test Movie",
            year=2024
        )

        assert isinstance(record, Title)
        assert record.record_id == "tmdb_123"
        assert record.title == "Test Movie"
        assert record.year == 2024
        assert record.dataset_name == "tmdb"  # Title's default

    def test_file_storage_with_title_class(self):
        """Test FileStorage creates Title objects when configured."""
        # Create a temporary JSON file with title data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {
                    "record_id": "tmdb_001",
                    "title": "Movie One",
                    "year": 2023
                },
                {
                    "record_id": "tmdb_002",
                    "title": "Movie Two",
                    "year": 2024
                }
            ], f)
            temp_file = f.name

        try:
            # Create FileStorage with Title class
            config = FileStorageConfig(
                type="file",
                path=temp_file,
                dataset_name="tmdb_test",
                record_class="buttermilk.tools.catalog_test.Title"
            )

            storage = FileStorage(config)

            # Iterate and verify Title objects are created
            records = list(storage)
            assert len(records) == 2

            # Check first record
            assert isinstance(records[0], Title)
            assert records[0].title == "Movie One"
            assert records[0].year == 2023

            # Check second record
            assert isinstance(records[1], Title)
            assert records[1].title == "Movie Two"
            assert records[1].year == 2024

        finally:
            # Clean up
            Path(temp_file).unlink()

    def test_file_storage_with_default_record_class(self):
        """Test FileStorage creates Record objects by default."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {
                    "record_id": "001",
                    "content": "Test content 1",
                    "dataset_name": "test"
                },
                {
                    "record_id": "002",
                    "content": "Test content 2",
                    "dataset_name": "test"
                }
            ], f)
            temp_file = f.name

        try:
            # Create FileStorage without specifying record_class
            config = FileStorageConfig(
                type="file",
                path=temp_file,
                dataset_name="test"
            )

            storage = FileStorage(config)

            # Iterate and verify Record objects are created
            records = list(storage)
            assert len(records) == 2

            # Check that they are Record instances, not Title
            assert isinstance(records[0], Record)
            assert not isinstance(records[0], Title)
            assert records[0].content == "Test content 1"

            assert isinstance(records[1], Record)
            assert not isinstance(records[1], Title)
            assert records[1].content == "Test content 2"

        finally:
            # Clean up
            Path(temp_file).unlink()

    def test_record_class_caching(self):
        """Test that record class is cached after first resolution."""
        config = BaseStorageConfig(
            type="test",
            dataset_name="test",
            record_class="buttermilk.tools.catalog_test.Title"
        )

        class TestStorage(Storage):
            def __iter__(self):
                return iter([])
            def save(self, records):
                pass
            def count(self):
                return 0

        storage = TestStorage(config)

        # First call should import and cache
        class1 = storage._get_record_class()
        assert class1 is Title

        # Second call should return cached value
        with patch('buttermilk.storage.base.importlib.import_module') as mock_import:
            class2 = storage._get_record_class()
            assert class2 is Title
            # import_module should not be called since class is cached
            mock_import.assert_not_called()