"""Test logger configuration validation for early failure detection."""

import pytest
from unittest.mock import Mock, patch

from buttermilk._core.bm_init import BM
from buttermilk._core.config import LoggerConfig
from pydantic import ValidationError


class TestLoggerConfigValidation:
    """Test that logger configuration is validated early and with clear error messages."""
    
    def test_valid_gcp_logger_config(self):
        """Test that valid GCP logger config passes validation."""
        config = {
            "platform": "local",
            "name": "test",
            "job": "testing",
            "logger_cfg": {
                "type": "gcp",
                "project": "test-project",
                "location": "us-central1"
            }
        }
        
        # Should not raise any validation errors
        bm = BM(**config)
        assert bm.logger_cfg is not None
        assert bm.logger_cfg.type == "gcp"
        assert bm.logger_cfg.project == "test-project"
        assert bm.logger_cfg.location == "us-central1"
    
    def test_gcp_logger_missing_project_fails_early(self):
        """Test that GCP logger config without project fails during initialization."""
        config = {
            "logger_cfg": {
                "type": "gcp",
                # Missing project field
                "location": "us-central1"
            }
        }
        
        # Should raise validation error early
        with pytest.raises(ValidationError) as exc_info:
            BM(**config)
        
        # Check error message is helpful
        assert "GCP logger configuration requires these fields: project" in str(exc_info.value)
    
    def test_gcp_logger_missing_location_fails_early(self):
        """Test that GCP logger config without location fails during initialization."""
        config = {
            "logger_cfg": {
                "type": "gcp",
                "project": "test-project"
                # Missing location field
            }
        }
        
        # Should raise validation error early
        with pytest.raises(ValidationError) as exc_info:
            BM(**config)
        
        # Check error message is helpful
        assert "GCP logger configuration requires these fields: location" in str(exc_info.value)
    
    def test_gcp_logger_missing_both_fields_fails_early(self):
        """Test that GCP logger config without both fields fails with clear message."""
        config = {
            "logger_cfg": {
                "type": "gcp"
                # Missing both project and location
            }
        }
        
        # Should raise validation error early
        with pytest.raises(ValidationError) as exc_info:
            BM(**config)
        
        # Check error message mentions both fields
        error_msg = str(exc_info.value)
        assert "project" in error_msg
        assert "location" in error_msg
    
    def test_unsupported_logger_type_fails_early(self):
        """Test that unsupported logger type fails during initialization."""
        config = {
            "logger_cfg": {
                "type": "azure",  # Not supported
                "project": "test-project",
                "location": "us-central1"
            }
        }
        
        # Should raise validation error early
        with pytest.raises(ValidationError) as exc_info:
            BM(**config)
        
        # Check error message is helpful
        assert "Unsupported logger type: 'azure'" in str(exc_info.value)
        assert "Supported logger types are: 'gcp', 'local'" in str(exc_info.value)
    
    def test_local_logger_config_no_requirements(self):
        """Test that local logger config doesn't require project or location."""
        config = {
            "platform": "local",
            "name": "test",
            "job": "testing",
            "logger_cfg": {
                "type": "local"
                # No other fields required
            }
        }
        
        # Should not raise any validation errors
        bm = BM(**config)
        assert bm.logger_cfg is not None
        assert bm.logger_cfg.type == "local"
    
    @patch('buttermilk._core.bm_init.logger')
    def test_cloud_service_unavailable_clear_error(self, mock_logger):
        """Test that cloud service unavailability gives clear error message."""
        config = {
            "platform": "local",
            "name": "test",
            "job": "testing",
            "logger_cfg": {
                "type": "gcp",
                "project": "test-project",
                "location": "us-central1"
            }
        }
        
        # Mock cloud manager to simulate service unavailable
        with patch('buttermilk._core.cloud.CloudManager') as mock_cloud_manager:
            mock_cm_instance = Mock()
            mock_cm_instance.gcs_log_client.side_effect = Exception("Service unavailable")
            mock_cloud_manager.return_value = mock_cm_instance
            
            # Create BM instance - should not fail during init
            bm = BM(**config)
            
            # Setup logging should handle the error gracefully
            bm.setup_logging()
            
            # Should log a warning about service unavailability, not config issue
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            assert any("service may be unavailable" in str(call) for call in warning_calls)
            assert any("Continuing with local logging only" in str(call) for call in warning_calls)
    
    def test_logger_config_direct_validation(self):
        """Test LoggerConfig validation directly."""
        # Valid GCP config
        valid_config = LoggerConfig(
            type="gcp",
            project="test-project",
            location="us-central1"
        )
        assert valid_config.project == "test-project"
        
        # Invalid GCP config - missing project
        with pytest.raises(ValidationError) as exc_info:
            LoggerConfig(
                type="gcp",
                location="us-central1"
            )
        assert "project" in str(exc_info.value)
        
        # Invalid GCP config - missing location
        with pytest.raises(ValidationError) as exc_info:
            LoggerConfig(
                type="gcp",
                project="test-project"
            )
        assert "location" in str(exc_info.value)
    
    def test_cloud_logging_fallback_indication(self):
        """Test that fallback to local logging is clearly indicated."""
        config = {
            "platform": "local",
            "name": "test",
            "job": "testing",
            "logger_cfg": {
                "type": "gcp",
                "project": "test-project",
                "location": "us-central1"
            }
        }
        
        # Mock to simulate cloud logging setup failure
        with patch('google.cloud.logging_v2.handlers.CloudLoggingHandler') as mock_handler:
            mock_handler.side_effect = Exception("Authentication failed")
            
            with patch('buttermilk._core.bm_init.logger') as mock_logger:
                bm = BM(**config)
                bm._cloud_manager = Mock()  # Ensure cloud manager exists
                
                # This should not raise but should log warnings
                bm.setup_logging()
                
                # Check that warning was logged about fallback
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                assert any("Continuing with local logging only" in str(call) for call in warning_calls)