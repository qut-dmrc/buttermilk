"""Comprehensive tests for FileStorage column mapping functionality.

This test suite validates the column mapping system in FileStorage, specifically
focusing on nested metadata mapping which is required for OSB data loading.

Tests cover:
1. Simple flat column mappings (existing functionality)
2. Nested metadata mappings (OSB use case)
3. Mixed mappings (combination of direct and nested)
4. Edge cases and error conditions
5. Real OSB data structure validation

These tests ensure that the column mapping bug fix works correctly while 
maintaining backward compatibility with existing configurations.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from buttermilk._core.storage_config import StorageConfig
from buttermilk._core.types import Record
from buttermilk.storage.file import FileStorage


class TestFileStorageColumnMapping:
    """Test FileStorage column mapping functionality."""
    
    def create_test_storage(self, config_dict: dict[str, Any]) -> FileStorage:
        """Create FileStorage instance with test configuration.
        
        Args:
            config_dict: Storage configuration dictionary
            
        Returns:
            FileStorage instance configured for testing
        """
        # Create temporary file for testing
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_dict['path'] = temp_file.name
        
        config = StorageConfig(**config_dict)
        storage = FileStorage(config)
        
        return storage
    
    def create_test_data_file(self, data: list[dict], file_path: str) -> None:
        """Create test JSON file with provided data.
        
        Args:
            data: List of dictionaries to write to file
            file_path: Path to write file to
        """
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def test_simple_column_mapping(self):
        """Test simple flat column mapping (existing functionality).
        
        This test validates that simple field renaming works correctly,
        which is the foundation for more complex mappings.
        """
        # Configuration similar to tox.yaml
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {
                'content': 'alt_text',
                'ground_truth': 'expected',
                'record_id': 'id'
            }
        })
        
        # Test data
        test_data = [
            {
                'id': 'test-001',
                'alt_text': 'This is the main content',
                'expected': {'toxicity': 0.8},
                'other_field': 'should be preserved as metadata'
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        # Load and validate
        records = list(storage)
        assert len(records) == 1
        
        record = records[0]
        assert record.record_id == 'test-001'
        assert record.content == 'This is the main content'
        assert record.ground_truth == {'toxicity': 0.8}
        # Other field should be preserved in metadata
        assert 'other_field' in record.metadata
        assert record.metadata['other_field'] == 'should be preserved as metadata'
    
    def test_nested_metadata_mapping(self):
        """Test nested metadata mapping (OSB use case).
        
        This test validates the core functionality needed for OSB data loading
        where multiple source fields need to be mapped into a metadata structure.
        """
        # Configuration similar to osb.yaml
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {
                'record_id': 'record_id',
                'content': 'fulltext',
                'metadata': {
                    'title': 'title',
                    'description': 'content',  # Different from fulltext!
                    'result': 'result',
                    'case_type': 'type',
                    'location': 'location',
                    'topics': 'topics',
                    'standards': 'standards'
                }
            }
        })
        
        # Test data matching OSB structure
        test_data = [
            {
                'record_id': 'OSB-TEST-001',
                'fulltext': 'This is the main fulltext content for vector processing...',
                'title': 'Test Case Title',
                'content': 'This is the case description content',  # Different from fulltext
                'result': 'upheld',
                'type': 'summary',
                'location': 'Ukraine',
                'topics': ['War and conflict', 'Content moderation'],
                'standards': ['Community Standards', 'Human Rights'],
                'unmapped_field': 'This should be preserved'
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        # Load and validate
        records = list(storage)
        assert len(records) == 1
        
        record = records[0]
        
        # Validate direct field mappings
        assert record.record_id == 'OSB-TEST-001'
        assert record.content == 'This is the main fulltext content for vector processing...'
        
        # Validate nested metadata mapping
        expected_metadata = {
            'title': 'Test Case Title',
            'description': 'This is the case description content',
            'result': 'upheld',
            'case_type': 'summary',
            'location': 'Ukraine',
            'topics': ['War and conflict', 'Content moderation'],
            'standards': ['Community Standards', 'Human Rights']
        }
        
        for key, expected_value in expected_metadata.items():
            assert key in record.metadata, f"Missing metadata key: {key}"
            assert record.metadata[key] == expected_value, f"Metadata mismatch for {key}"
        
        # Validate that unmapped fields are preserved
        assert 'unmapped_field' in record.metadata
        assert record.metadata['unmapped_field'] == 'This should be preserved'
    
    def test_mixed_mapping(self):
        """Test mixed mapping (direct fields + nested metadata).
        
        This test validates that combinations of direct field mapping
        and nested metadata mapping work correctly together.
        """
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {
                'content': 'text',
                'ground_truth': 'expected',
                'metadata': {
                    'title': 'title',
                    'summary': 'description'
                }
            }
        })
        
        test_data = [
            {
                'text': 'Main content text',
                'expected': {'score': 0.5},
                'title': 'Document Title',
                'description': 'Document summary',
                'extra_field': 'Should be preserved'
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        records = list(storage)
        record = records[0]
        
        # Validate direct mappings
        assert record.content == 'Main content text'
        assert record.ground_truth == {'score': 0.5}
        
        # Validate nested metadata
        assert record.metadata['title'] == 'Document Title'
        assert record.metadata['summary'] == 'Document summary'
        
        # Validate preserved fields
        assert record.metadata['extra_field'] == 'Should be preserved'
    
    def test_real_osb_data_structure(self):
        """Test with realistic OSB data structure.
        
        This test uses a data structure that closely matches the actual
        OSB JSON format to ensure the mapping works in practice.
        """
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {
                'record_id': 'record_id',
                'content': 'fulltext',
                'metadata': {
                    'title': 'title',
                    'description': 'content',
                    'result': 'result',
                    'type': 'type',
                    'location': 'location',
                    'case_date': 'case_date',
                    'topics': 'topics',
                    'standards': 'standards',
                    'reasons': 'reasons',
                    'recommendations': 'recommendations',
                    'job_id': 'job_id',
                    'timestamp': 'timestamp'
                }
            }
        })
        
        # Realistic OSB data structure
        test_data = [
            {
                'record_id': 'BUN-QBBLZ8WI',
                'fulltext': '## Case Summary\n\nA user appealed Meta\'s decision...',
                'title': 'Mention of Al-Shabaab',
                'content': 'The first post included a picture showing weapons...',
                'result': 'leave up',
                'type': 'summary',
                'location': 'Somalia',
                'case_date': '2023-11-22',
                'topics': ['War and conflict', 'Dangerous individuals and organizations'],
                'standards': ['Dangerous Individuals and Organizations policy'],
                'reasons': [
                    'The policy prohibits content that "praises" dangerous organizations...',
                    'In the first post, the caption describes a military operation...'
                ],
                'recommendations': [
                    'Enhance training and accuracy of reviewers...',
                    'Add criteria and illustrative examples...'
                ],
                'job_id': '2Luac3REAVKPnF52dZqtc4',
                'timestamp': 1732052347313
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        records = list(storage)
        record = records[0]
        
        # Validate structure matches expectations
        assert record.record_id == 'BUN-QBBLZ8WI'
        assert record.content.startswith('## Case Summary')
        
        # Validate all metadata fields are correctly mapped
        assert record.metadata['title'] == 'Mention of Al-Shabaab'
        assert record.metadata['description'] == 'The first post included a picture showing weapons...'
        assert record.metadata['result'] == 'leave up'
        assert record.metadata['type'] == 'summary'
        assert record.metadata['location'] == 'Somalia'
        assert record.metadata['case_date'] == '2023-11-22'
        
        # Validate array fields are preserved
        assert isinstance(record.metadata['topics'], list)
        assert len(record.metadata['topics']) == 2
        assert 'War and conflict' in record.metadata['topics']
        
        assert isinstance(record.metadata['standards'], list)
        assert len(record.metadata['standards']) == 1
        
        assert isinstance(record.metadata['reasons'], list)
        assert len(record.metadata['reasons']) == 2
        
        assert isinstance(record.metadata['recommendations'], list)
        assert len(record.metadata['recommendations']) == 2
        
        # Validate other fields
        assert record.metadata['job_id'] == '2Luac3REAVKPnF52dZqtc4'
        assert record.metadata['timestamp'] == 1732052347313
    
    def test_no_column_mapping(self):
        """Test behavior when no column mapping is specified.
        
        This test ensures that when columns={} or columns=None,
        the data is loaded with default field extraction.
        """
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {}  # No mapping
        })
        
        test_data = [
            {
                'record_id': 'test-001',
                'content': 'Direct content field',
                'title': 'Document Title',
                'other_field': 'Should go to metadata'
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        records = list(storage)
        record = records[0]
        
        # Should use direct field names
        assert record.record_id == 'test-001'
        assert record.content == 'Direct content field'
        
        # Other fields should be preserved as metadata
        assert 'title' in record.metadata
        assert record.metadata['title'] == 'Document Title'
        assert record.metadata['other_field'] == 'Should go to metadata'
    
    def test_missing_mapped_fields(self):
        """Test behavior when mapped source fields are missing.
        
        This test ensures graceful handling when configured field mappings
        reference fields that don't exist in the source data.
        """
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {
                'content': 'missing_field',  # This field doesn't exist
                'metadata': {
                    'title': 'existing_title',
                    'description': 'missing_description'  # This doesn't exist
                }
            }
        })
        
        test_data = [
            {
                'record_id': 'test-001',
                'existing_title': 'This exists',
                'other_field': 'Should be preserved',
                'text': 'Fallback content'  # Provide fallback since content field is missing
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        records = list(storage)
        record = records[0]
        
        # Missing content field should default to empty or fallback
        assert hasattr(record, 'content')
        
        # Existing metadata mapping should work
        assert record.metadata['title'] == 'This exists'
        
        # Missing metadata field should be absent (not create empty entries)
        assert 'description' not in record.metadata or record.metadata['description'] is None
        
        # Other fields should be preserved
        assert record.metadata['other_field'] == 'Should be preserved'
    
    def test_empty_metadata_mapping(self):
        """Test behavior with empty metadata mapping.
        
        This test validates that empty metadata configurations don't break
        the system and that data is handled gracefully.
        """
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {
                'content': 'text',
                'metadata': {}  # Empty metadata mapping
            }
        })
        
        test_data = [
            {
                'text': 'Main content',
                'title': 'Document Title',
                'other_field': 'Should be preserved'
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        records = list(storage)
        record = records[0]
        
        assert record.content == 'Main content'
        
        # Fields should still be preserved as metadata
        assert 'title' in record.metadata
        assert record.metadata['title'] == 'Document Title'
        assert record.metadata['other_field'] == 'Should be preserved'


class TestFileStorageRegressionTests:
    """Regression tests to ensure existing functionality isn't broken."""
    
    def create_test_storage(self, config_dict: dict[str, Any]) -> FileStorage:
        """Create FileStorage instance with test configuration."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_dict['path'] = temp_file.name
        
        config = StorageConfig(**config_dict)
        storage = FileStorage(config)
        
        return storage
    
    def create_test_data_file(self, data: list[dict], file_path: str) -> None:
        """Create test JSON file with provided data."""
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    def test_tox_config_compatibility(self):
        """Test that existing tox.yaml configuration still works.
        
        This regression test ensures that the column mapping fix doesn't
        break existing simple column mappings used in tox configuration.
        """
        # Configuration matching tox.yaml (drag dataset)
        storage = self.create_test_storage({
            'type': 'file',
            'columns': {
                'content': 'alt_text',
                'ground_truth': 'expected',
                'title': 'name'
            }
        })
        
        test_data = [
            {
                'alt_text': 'This is toxic content',
                'expected': {'toxicity': 0.9, 'severe_toxicity': 0.1},
                'name': 'Toxic comment example',
                'extra_field': 'Should be preserved'
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        records = list(storage)
        record = records[0]
        
        # Validate simple mappings work
        assert record.content == 'This is toxic content'
        assert record.ground_truth == {'toxicity': 0.9, 'severe_toxicity': 0.1}
        
        # Title should be in metadata (since it's not a Record field)
        assert 'title' in record.metadata
        assert record.metadata['title'] == 'Toxic comment example'
        
        # Extra fields should be preserved
        assert record.metadata['extra_field'] == 'Should be preserved'
    
    def test_tja_config_compatibility(self):
        """Test that existing tja_train.yaml configuration still works.
        
        This regression test ensures that configurations with no column
        mapping continue to work correctly.
        """
        # Configuration matching tja_train.yaml (no column mapping)
        storage = self.create_test_storage({
            'type': 'file'
            # No columns field - should use default behavior
        })
        
        test_data = [
            {
                'record_id': 'tja-001',
                'content': 'Journalism training content',
                'title': 'Training Article',
                'category': 'media_literacy',
                'author': 'Training Team'
            }
        ]
        
        self.create_test_data_file(test_data, storage.config.path)
        
        records = list(storage)
        record = records[0]
        
        # Should use fields directly when no mapping is specified
        assert record.record_id == 'tja-001'
        assert record.content == 'Journalism training content'
        
        # Other fields should be preserved as metadata
        assert record.metadata['title'] == 'Training Article'
        assert record.metadata['category'] == 'media_literacy'
        assert record.metadata['author'] == 'Training Team'


@pytest.mark.integration
class TestFileStorageIntegration:
    """Integration tests for FileStorage with different file formats."""
    
    def test_jsonl_format_with_column_mapping(self):
        """Test JSONL format with column mapping.
        
        This test ensures that the column mapping fix works correctly
        with JSONL (JSON Lines) format files as well as regular JSON.
        """
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        config = StorageConfig(**{
            'type': 'file',
            'path': temp_file.name,
            'columns': {
                'content': 'fulltext',
                'metadata': {
                    'title': 'title',
                    'summary': 'description'
                }
            }
        })
        
        storage = FileStorage(config)
        
        # Create JSONL test data (one JSON object per line)
        test_data = [
            {'fulltext': 'Content 1', 'title': 'Title 1', 'description': 'Desc 1'},
            {'fulltext': 'Content 2', 'title': 'Title 2', 'description': 'Desc 2'}
        ]
        
        with open(temp_file.name, 'w') as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
        
        # Load and validate
        records = list(storage)
        assert len(records) == 2
        
        for i, record in enumerate(records):
            assert record.content == f'Content {i+1}'
            assert record.metadata['title'] == f'Title {i+1}'
            assert record.metadata['summary'] == f'Desc {i+1}'
    
    def test_cloud_path_simulation(self):
        """Test that cloud paths work with column mapping.
        
        This test simulates cloud storage paths (GCS) to ensure that
        the column mapping works correctly regardless of storage location.
        """
        # Note: This is a simulation test since we can't easily test real GCS in unit tests
        # The actual cloud path handling is tested separately
        
        # Create local file that simulates cloud data structure
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        config = StorageConfig(**{
            'type': 'file',  # FileStorage handles both local and cloud paths
            'path': temp_file.name,  # In real scenario this would be gs://...
            'columns': {
                'record_id': 'record_id',
                'content': 'fulltext',
                'metadata': {
                    'title': 'title',
                    'case_type': 'type',
                    'result': 'result'
                }
            }
        })
        
        storage = FileStorage(config)
        
        # Simulate OSB-like cloud data
        test_data = [
            {
                'record_id': 'CLOUD-001',
                'fulltext': 'Cloud-stored content',
                'title': 'Cloud Test Case',
                'type': 'summary',
                'result': 'upheld'
            }
        ]
        
        with open(temp_file.name, 'w') as f:
            json.dump(test_data, f)
        
        records = list(storage)
        record = records[0]
        
        assert record.record_id == 'CLOUD-001'
        assert record.content == 'Cloud-stored content'
        assert record.metadata['title'] == 'Cloud Test Case'
        assert record.metadata['case_type'] == 'summary'
        assert record.metadata['result'] == 'upheld'