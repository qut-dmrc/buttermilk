"""Test BM GenAI client integration.

This test verifies that:
1. GenAI client is properly exposed through BM.genai
2. It's initialized with correct Vertex AI parameters
3. Error handling works correctly
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from buttermilk._core.bm_init import BM
from buttermilk._core.config import CloudProviderCfg


@pytest.fixture
def mock_cloud_config():
    """Create mock cloud configuration with location."""
    return CloudProviderCfg(
        type="gcp",
        project_id="test-project",
        location="us-central1",
    )


@pytest.fixture
def bm_config(mock_cloud_config):
    """Create BM configuration."""
    return {
        "run_info": {
            "name": "test",
            "job": "test_job",
        },
        "clouds": [mock_cloud_config],
    }


@patch("buttermilk._core.cloud.genai")
@patch("buttermilk._core.cloud.default")
def test_genai_client_initialization(mock_default, mock_genai, bm_config):
    """Test that GenAI client is initialized with correct parameters."""
    # Mock GCP credentials
    mock_creds = Mock()
    mock_creds.valid = True
    mock_creds.token = "test-token"
    mock_default.return_value = (mock_creds, "test-project")
    
    # Mock GenAI client
    mock_genai_client = Mock()
    mock_genai.Client.return_value = mock_genai_client
    
    # Create BM instance
    bm = BM(**bm_config)
    
    # Access genai property
    genai_client = bm.genai
    
    # Verify GenAI client was created with correct parameters
    mock_genai.Client.assert_called_once_with(
        vertex=True,
        project="test-project",
        location="us-central1",
    )
    
    # Verify we got the mock client back
    assert genai_client == mock_genai_client


@patch("buttermilk._core.cloud.genai")
@patch("buttermilk._core.cloud.default")
def test_genai_client_cached_property(mock_default, mock_genai, bm_config):
    """Test that GenAI client is created only once (cached)."""
    # Mock GCP credentials
    mock_creds = Mock()
    mock_creds.valid = True
    mock_creds.token = "test-token"
    mock_default.return_value = (mock_creds, "test-project")
    
    # Mock GenAI client
    mock_genai_client = Mock()
    mock_genai.Client.return_value = mock_genai_client
    
    # Create BM instance
    bm = BM(**bm_config)
    
    # Access genai property multiple times
    client1 = bm.genai
    client2 = bm.genai
    client3 = bm.genai
    
    # Verify GenAI client was created only once
    mock_genai.Client.assert_called_once()
    
    # Verify same instance returned
    assert client1 is client2
    assert client2 is client3


def test_genai_client_missing_location():
    """Test error when location is missing from config."""
    # Create config without location
    cloud_config = CloudProviderCfg(
        type="gcp",
        project_id="test-project",
        # location missing
    )
    
    bm_config = {
        "run_info": {
            "name": "test",
            "job": "test_job",
        },
        "clouds": [cloud_config],
    }
    
    with patch("buttermilk._core.cloud.default") as mock_default:
        mock_creds = Mock()
        mock_creds.valid = True
        mock_creds.token = "test-token"
        mock_default.return_value = (mock_creds, "test-project")
        
        bm = BM(**bm_config)
        
        # Accessing genai should raise error about missing location
        with pytest.raises(RuntimeError, match="GCP location not specified"):
            _ = bm.genai


def test_genai_client_no_gcp_config():
    """Test error when no GCP config is present."""
    bm_config = {
        "run_info": {
            "name": "test",
            "job": "test_job",
        },
        "clouds": [],  # No cloud configs
    }
    
    bm = BM(**bm_config)
    
    # Accessing genai should raise error about missing config
    with pytest.raises(RuntimeError, match="No GCP cloud configuration found"):
        _ = bm.genai