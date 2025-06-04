"""Test parameter merging behavior in AgentVariants.get_configs method."""

import pytest
from buttermilk._core.config import AgentVariants
from buttermilk._core.types import RunRequest


def test_parameter_merging_string_overrides_list():
    """Test that params.parameters (string) overrides flow_default_parameters (list)."""
    # Setup
    flow_default_parameters = {
        'criteria': ['trans_simplified', 'cte', 'tja', 'glaad', 'hrc', 'trans_factored']
    }
    
    params_parameters = {
        'criteria': 'glaad'
    }
    
    # Test direct merging behavior
    final_params = {**flow_default_parameters, **params_parameters}
    
    assert final_params['criteria'] == 'glaad'
    assert isinstance(final_params['criteria'], str)
    
    
def test_agent_variants_parameter_merging():
    """Test actual AgentVariants.get_configs parameter merging."""
    # Create a minimal AgentVariants config
    agent_variants = AgentVariants(
        role="TEST_AGENT",
        agent_obj="LLMAgent",  # This needs to be registered, but we'll mock that
        parameters={}
    )
    
    # Create RunRequest with string criteria
    run_request = RunRequest(
        flow="test_flow",
        ui_type="test",
        parameters={'criteria': 'glaad'}
    )
    
    # Flow default parameters with list criteria
    flow_default_parameters = {
        'criteria': ['trans_simplified', 'cte', 'tja', 'glaad', 'hrc', 'trans_factored']
    }
    
    # Mock the AgentRegistry to avoid registration issues
    from unittest.mock import patch, MagicMock
    
    mock_agent_class = MagicMock()
    
    with patch('buttermilk._core.variants.AgentRegistry') as mock_registry:
        mock_registry.get.return_value = mock_agent_class
        
        configs = agent_variants.get_configs(
            params=run_request, 
            flow_default_params=flow_default_parameters
        )
        
        # Should generate one config
        assert len(configs) == 1
        
        agent_class, agent_config = configs[0]
        assert agent_class == mock_agent_class
        
        # The key test: criteria should be the string 'glaad', not the list
        assert agent_config.parameters['criteria'] == 'glaad'
        assert isinstance(agent_config.parameters['criteria'], str)


def test_manual_merge_order():
    """Test the exact merge order used in get_configs method."""
    flow_default_params = {'criteria': ['list', 'values']}
    base_parameters = {'criteria': 'string_value'}
    parallel_params = {}
    task_params = {}
    
    # This is the exact line from get_configs:771
    final_params = {**flow_default_params, **base_parameters, **parallel_params, **task_params}
    
    # base_parameters should win over flow_default_params
    assert final_params['criteria'] == 'string_value'
    assert isinstance(final_params['criteria'], str)


def test_variant_filtering_with_runrequest_override():
    """Test that variants are filtered when RunRequest parameters override them."""
    # Create AgentVariants with criteria in variants
    agent_variants = AgentVariants(
        role="TEST_AGENT",
        agent_obj="LLMAgent",
        parameters={},
        variants={'criteria': ['variant1', 'variant2', 'variant3']}  # This should be filtered out
    )
    
    # Create RunRequest that overrides criteria
    run_request = RunRequest(
        flow="test_flow",
        ui_type="test", 
        parameters={'criteria': 'glaad_override'}  # This should win
    )
    
    # Mock the AgentRegistry
    from unittest.mock import patch, MagicMock
    mock_agent_class = MagicMock()
    
    with patch('buttermilk._core.variants.AgentRegistry') as mock_registry:
        mock_registry.get.return_value = mock_agent_class
        
        configs = agent_variants.get_configs(
            params=run_request,
            flow_default_params={}
        )
        
        # Should generate only one config (not 3 from the variants)
        assert len(configs) == 1
        
        agent_class, agent_config = configs[0]
        assert agent_class == mock_agent_class
        
        # The criteria should be the RunRequest override, not from variants
        assert agent_config.parameters['criteria'] == 'glaad_override'
        assert isinstance(agent_config.parameters['criteria'], str)