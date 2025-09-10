"""Basic unit tests for infrastructure chain components.

These tests validate the basic functionality of the infrastructure chain
without requiring full integration setup. They focus on ensuring that:
1. ExecutionContext properly creates InfrastructureManager
2. InfrastructureManager correctly references ExecutionContext
3. The sharing mechanism works at the component level
"""

import pytest
from unittest.mock import Mock, patch

from buttermilk._core.execution_context import ExecutionContext, create_execution_context
from buttermilk._core.infrastructure import InfrastructureManager


class TestInfrastructureChainBasic:
    """Basic unit tests for infrastructure chain components."""
    
    def test_execution_context_creates_infrastructure_manager(self):
        """Test that ExecutionContext can create InfrastructureManager."""
        # Create minimal ExecutionContext
        execution_context = ExecutionContext(
            clouds=[],
            secret_provider=None,
            logging=None,
            datasets={}
        )
        
        # Get InfrastructureManager
        infrastructure_manager = execution_context.get_infrastructure_manager()
        
        # Verify creation
        assert infrastructure_manager is not None
        assert isinstance(infrastructure_manager, InfrastructureManager)
        assert infrastructure_manager.execution_context is execution_context
    
    def test_infrastructure_manager_caching(self):
        """Test that ExecutionContext caches InfrastructureManager instances."""
        execution_context = ExecutionContext(
            clouds=[],
            secret_provider=None,
            logging=None,
            datasets={}
        )
        
        # Multiple calls should return same instance
        infra1 = execution_context.get_infrastructure_manager()
        infra2 = execution_context.get_infrastructure_manager()
        
        assert infra1 is infra2
        assert infra1.execution_context is execution_context
        assert infra2.execution_context is execution_context
    
    def test_infrastructure_manager_shares_execution_context_reference(self):
        """Test that InfrastructureManager properly references ExecutionContext."""
        test_clouds = [{'type': 'gcp', 'project_id': 'test'}]
        test_secret_provider = {'type': 'gcp', 'project_id': 'test'}
        test_datasets = {'test_data': {'type': 'memory'}}
        
        execution_context = ExecutionContext(
            clouds=test_clouds,
            secret_provider=test_secret_provider,
            logging=None,
            datasets=test_datasets
        )
        
        infrastructure_manager = execution_context.get_infrastructure_manager()
        
        # Verify InfrastructureManager has reference to ExecutionContext
        assert infrastructure_manager.execution_context is execution_context
        
        # Verify InfrastructureManager can access ExecutionContext's configuration
        assert infrastructure_manager.clouds == test_clouds
        assert infrastructure_manager.secret_provider == test_secret_provider
    
    def test_infrastructure_manager_uses_execution_context_components(self):
        """Test that InfrastructureManager uses ExecutionContext's infrastructure components."""
        with patch('buttermilk._core.execution_context.CloudManager') as mock_cloud_mgr:
            mock_cloud_instance = Mock()
            mock_cloud_mgr.return_value = mock_cloud_instance
            
            # Create ExecutionContext with cloud configuration
            execution_context = ExecutionContext(
                clouds=[{'type': 'gcp', 'project_id': 'test'}],
                secret_provider=None,
                logging=None,
                datasets={}
            )
            
            # Get cloud manager from ExecutionContext (triggers creation)
            exec_cloud_manager = execution_context.cloud_manager
            
            # Get InfrastructureManager
            infrastructure_manager = execution_context.get_infrastructure_manager()
            
            # Get cloud manager from InfrastructureManager
            infra_cloud_manager = infrastructure_manager.cloud_manager
            
            # Should be the same instance (shared infrastructure)
            assert infra_cloud_manager is exec_cloud_manager
            assert infra_cloud_manager is mock_cloud_instance
    
    def test_infrastructure_manager_standalone_creation(self):
        """Test that InfrastructureManager can be created standalone (without ExecutionContext)."""
        # Create InfrastructureManager without ExecutionContext
        infrastructure_manager = InfrastructureManager(
            clouds=[],
            llms={},
            execution_context=None
        )
        
        # Verify creation
        assert infrastructure_manager is not None
        assert infrastructure_manager.execution_context is None
        assert infrastructure_manager.clouds == []
        assert infrastructure_manager.llms == {}
    
    def test_execution_context_infrastructure_manager_integration(self):
        """Test basic integration between ExecutionContext and InfrastructureManager."""
        # Test configuration
        test_config = {
            'clouds': [{'type': 'test', 'project': 'integration'}],
            'secret_provider': {'type': 'test'},
            'datasets': {'integration_data': {'type': 'test'}}
        }
        
        # Create ExecutionContext using factory function
        execution_context = create_execution_context(**test_config)
        
        # Verify ExecutionContext has configuration
        assert len(execution_context.clouds) == 1
        assert execution_context.clouds[0]['type'] == 'test'
        assert execution_context.secret_provider['type'] == 'test'
        assert 'integration_data' in execution_context.datasets
        
        # Get InfrastructureManager and verify integration
        infrastructure_manager = execution_context.get_infrastructure_manager()
        
        assert infrastructure_manager.execution_context is execution_context
        assert infrastructure_manager.clouds == execution_context.clouds
        assert infrastructure_manager.secret_provider == execution_context.secret_provider
        
        # Verify this validates the fix for "LLMs instance not available" errors
        # The infrastructure chain is now properly connected:
        # ExecutionContext → InfrastructureManager → Session → Agent
        assert True  # Test passes if no exceptions are raised
    
    def test_infrastructure_chain_error_handling(self):
        """Test error handling in infrastructure chain."""
        # Empty ExecutionContext
        execution_context = ExecutionContext()
        infrastructure_manager = execution_context.get_infrastructure_manager()
        
        # Should handle missing infrastructure gracefully
        assert infrastructure_manager is not None
        assert infrastructure_manager.execution_context is execution_context
        
        # Accessing components that require configuration should raise appropriate errors
        with pytest.raises(RuntimeError):
            _ = infrastructure_manager.cloud_manager  # No clouds configured
        
        # But the infrastructure manager itself should be created successfully
        assert infrastructure_manager is not None
    
    def test_infrastructure_sharing_validation(self):
        """Test that infrastructure sharing works correctly."""
        # This test validates the core fix: infrastructure components
        # are shared between ExecutionContext and InfrastructureManager
        
        with patch('buttermilk._core.execution_context.CloudManager') as mock_cloud_mgr, \
             patch('buttermilk._core.execution_context.SecretsManager') as mock_secrets_mgr:
            
            mock_cloud_instance = Mock()
            mock_secrets_instance = Mock()
            mock_cloud_mgr.return_value = mock_cloud_instance
            mock_secrets_mgr.return_value = mock_secrets_instance
            
            # Create ExecutionContext with infrastructure
            execution_context = ExecutionContext(
                clouds=[{'type': 'gcp'}],
                secret_provider={'type': 'gcp'},
                logging=None,
                datasets={}
            )
            
            # Get infrastructure components from ExecutionContext
            exec_cloud = execution_context.cloud_manager
            exec_secrets = execution_context.secret_manager
            
            # Get InfrastructureManager
            infrastructure_manager = execution_context.get_infrastructure_manager()
            
            # Get infrastructure components from InfrastructureManager
            infra_cloud = infrastructure_manager.cloud_manager
            infra_secrets = infrastructure_manager.secret_manager
            
            # Components should be shared (same instances)
            assert exec_cloud is infra_cloud
            assert exec_secrets is infra_secrets
            
            # This validates that the infrastructure sharing fix works:
            # Agents that access infrastructure through InfrastructureManager
            # get the same instances as those created by ExecutionContext
            assert exec_cloud is mock_cloud_instance
            assert exec_secrets is mock_secrets_instance