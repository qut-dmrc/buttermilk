"""Test for load_data regression - Cannot instantiate typing.Union."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.storage_config import FileStorageConfig, BigQueryStorageConfig
from buttermilk._core.config import DataSourceConfig


class ConcreteOrchestrator(Orchestrator):
    """Concrete implementation for testing."""
    
    async def _setup(self) -> None:
        """Setup implementation."""
        pass
    
    async def _cleanup(self) -> None:
        """Cleanup implementation."""
        pass
    
    async def _run(self, run_request) -> None:
        """Run implementation."""
        pass


class TestLoadDataRegression:
    """Test that load_data properly handles storage config conversion."""
    
    @pytest.mark.asyncio
    async def test_load_data_with_validated_storage_configs(self):
        """Test load_data with storage configs already validated by model_validator."""
        # Create orchestrator with storage configs already converted
        orchestrator = ConcreteOrchestrator(
            orchestrator="test",
            name="test_flow",
            storage={
                "file_storage": FileStorageConfig(
                    type="file",
                    path="/data/test.json"
                ),
                "bq_storage": BigQueryStorageConfig(
                    type="bigquery",
                    project_id="test-project",
                    dataset_id="test_dataset"
                )
            },
            agents={},
            observers={},
            parameters={}
        )
        
        # Mock bm.get_storage
        with patch('buttermilk._core.orchestrator.bm') as mock_bm:
            mock_storage = Mock()
            mock_bm.get_storage.return_value = mock_storage
            
            # Should not raise "Cannot instantiate typing.Union"
            await orchestrator.load_data()
            
            # Verify storage was created for each config
            assert mock_bm.get_storage.call_count == 2
            
            # Verify the configs passed were the actual subclasses
            calls = mock_bm.get_storage.call_args_list
            assert isinstance(calls[0][0][0], FileStorageConfig)
            assert isinstance(calls[1][0][0], BigQueryStorageConfig)
    
    @pytest.mark.asyncio
    async def test_load_data_with_legacy_datasource_config(self):
        """Test load_data converts DataSourceConfig properly."""
        # Create orchestrator with legacy DataSourceConfig
        orchestrator = ConcreteOrchestrator(
            orchestrator="test",
            name="test_flow",
            storage={
                "legacy": DataSourceConfig(
                    type="file",
                    path="/data/legacy.json"
                )
            },
            agents={},
            observers={},
            parameters={}
        )
        
        # Mock create_data_loader since DataSourceConfig uses legacy loader
        with patch('buttermilk.data.loaders.create_data_loader') as mock_create_loader:
            mock_loader = Mock()
            mock_create_loader.return_value = mock_loader
            
            # Should use legacy create_data_loader for DataSourceConfig
            await orchestrator.load_data()
            
            # Verify create_data_loader was called with DataSourceConfig
            assert mock_create_loader.call_count == 1
            
            # Verify the config passed was the DataSourceConfig
            config = mock_create_loader.call_args[0][0]
            assert isinstance(config, DataSourceConfig)
            assert config.path == "/data/legacy.json"
            
            # Verify loader was stored
            assert orchestrator._input_loaders["legacy"] == mock_loader
    
    @pytest.mark.asyncio
    async def test_load_data_with_dict_configs(self):
        """Test load_data handles dict configs from OmegaConf."""
        # Simulate what happens when configs come from OmegaConf
        orchestrator = ConcreteOrchestrator(
            orchestrator="test",
            name="test_flow",
            storage={
                "dict_storage": {
                    "type": "bigquery",
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                    "table_id": "test_table"
                }
            },
            agents={},
            observers={},
            parameters={}
        )
        
        # Override storage to be a dict (simulating OmegaConf not being converted)
        orchestrator.storage = {
            "dict_storage": {
                "type": "bigquery",
                "project_id": "test-project",
                "dataset_id": "test_dataset",
                "table_id": "test_table"
            }
        }
        
        # Mock bm.get_storage
        with patch('buttermilk._core.orchestrator.bm') as mock_bm:
            mock_storage = Mock()
            mock_bm.get_storage.return_value = mock_storage
            
            # Should use StorageFactory to create proper config
            await orchestrator.load_data()
            
            # Verify storage was created
            assert mock_bm.get_storage.call_count == 1
            
            # Verify the config was converted to BigQueryStorageConfig
            storage_config = mock_bm.get_storage.call_args[0][0]
            assert isinstance(storage_config, BigQueryStorageConfig)
            assert storage_config.project_id == "test-project"
            assert storage_config.dataset_id == "test_dataset"
    
    @pytest.mark.asyncio
    async def test_get_record_dataset_triggers_load_data(self):
        """Test that get_record_dataset calls load_data if needed."""
        orchestrator = ConcreteOrchestrator(
            orchestrator="test",
            name="test_flow",
            storage={
                "test_storage": FileStorageConfig(
                    type="file",
                    path="/data/test.json"
                )
            },
            agents={},
            observers={},
            parameters={}
        )
        
        # Mock the storage loader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))  # Empty iterator
        
        with patch('buttermilk._core.orchestrator.bm') as mock_bm:
            mock_bm.get_storage.return_value = mock_loader
            
            # Try to get a record - should trigger load_data
            with pytest.raises(Exception, match="Unable to find requested record"):
                await orchestrator.get_record_dataset("test-id")
            
            # Verify load_data was called (storage was created)
            assert mock_bm.get_storage.called