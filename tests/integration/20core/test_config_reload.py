"""Integration tests for dynamic configuration reload functionality.

Tests the complete workflow of GCS-mounted configuration reloading
without disrupting active sessions.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from buttermilk._core.types import RunRequest
from buttermilk.runner.flowrunner import FlowRunner


class TestConfigurationReload:
    """Test configuration reload functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary configuration directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create basic config structure
        config_yaml = temp_dir / "config.yaml"
        config_yaml.write_text("""
defaults:
  - _self_
  - run: api

flows:
  test_flow:
    _target_: buttermilk.orchestrators.groupchat.GroupChatOrchestrator
    name: test_flow
    agents: {}
    parameters: {}
""")
        
        # Create run config
        run_dir = temp_dir / "run"
        run_dir.mkdir()
        api_yaml = run_dir / "api.yaml"
        api_yaml.write_text("""
_target_: buttermilk.runner.flowrunner.FlowRunner
mode: api
ui: web
flows: ${flows}
""")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_config_snapshot_saving(self, temp_config_dir):
        """Test that configuration snapshots are saved during flow execution."""
        # Create a mock FlowRunner with test flows
        flow_runner = FlowRunner(
            flows={"test_flow": {"name": "test_flow", "parameters": {"test": "value"}}},
            mode="api"
        )
        
        # Create a test run request
        run_request = RunRequest(
            flow="test_flow",
            session_id="test_session_123",
            job_id="test_job_456",
            parameters={"param1": "value1"},
            inputs={"input1": "data1"}
        )
        
        # Call the config snapshot method
        flow_runner._save_config_snapshot(run_request)
        
        # Verify snapshot files were created
        expected_session_dir = Path(f"/tmp/runs/{run_request.session_id}")
        config_snapshot_dir = expected_session_dir / "config_snapshot"
        
        assert config_snapshot_dir.exists(), "Config snapshot directory should be created"
        
        # Check latest.json was created
        latest_file = config_snapshot_dir / "latest.json"
        assert latest_file.exists(), "latest.json should be created"
        
        # Verify latest.json content
        with open(latest_file) as f:
            latest_data = json.load(f)
        
        assert latest_data["flow_name"] == "test_flow"
        assert latest_data["session_id"] == "test_session_123"
        assert "config_file" in latest_data
        assert "timestamp" in latest_data
        
        # Check flow-specific config file was created
        config_file_path = Path(latest_data["config_file"])
        assert config_file_path.exists(), "Flow-specific config file should be created"
        
        # Verify flow config content
        with open(config_file_path) as f:
            config_data = json.load(f)
        
        assert config_data["flow_name"] == "test_flow"
        assert config_data["session_id"] == "test_session_123"
        assert config_data["job_id"] == "test_job_456"
        assert config_data["run_parameters"] == {"param1": "value1"}
        assert config_data["run_inputs"] == {"input1": "data1"}
        assert "flow_config" in config_data

    @patch("buttermilk.runner.flowrunner.hydra")
    @patch("buttermilk.runner.flowrunner.initialize_config_dir")
    async def test_reload_configurations_success(self, mock_initialize, mock_hydra, temp_config_dir):
        """Test successful configuration reload."""
        # Mock Hydra configuration loading
        mock_conf = Mock()
        mock_conf.run.flows = {
            "test_flow": {"name": "test_flow", "updated": True},
            "new_flow": {"name": "new_flow", "new": True}
        }
        
        mock_hydra.compose.return_value = mock_conf
        mock_hydra.core.global_hydra.GlobalHydra.instance.return_value.clear.return_value = None
        
        # Create FlowRunner with initial flows
        flow_runner = FlowRunner(
            flows={"test_flow": {"name": "test_flow", "updated": False}},
            mode="api"
        )
        
        # Mock the config directory path
        with patch("pathlib.Path.resolve", return_value=temp_config_dir):
            # Test reload
            result = await flow_runner.reload_configurations()
        
        # Verify result
        assert result["success"] is True
        assert "test_flow" in result["flows_loaded"]
        assert "new_flow" in result["flows_loaded"]
        assert "test_flow" in result["flows_updated"]
        assert len(result["flows_removed"]) == 0
        assert len(result["errors"]) == 0
        
        # Verify flows were updated
        assert "new_flow" in flow_runner.flows
        assert flow_runner.flows["test_flow"]["updated"] is True

    @patch("buttermilk.runner.flowrunner.hydra")
    async def test_reload_configurations_failure(self, mock_hydra):
        """Test configuration reload failure handling."""
        # Mock Hydra to raise an exception
        mock_hydra.core.global_hydra.GlobalHydra.instance.return_value.clear.side_effect = Exception("Config error")
        
        # Create FlowRunner
        flow_runner = FlowRunner(
            flows={"test_flow": {"name": "test_flow"}},
            mode="api"
        )
        
        # Store original flows for rollback verification
        original_flows = flow_runner.flows.copy()
        
        # Test reload with error
        result = await flow_runner.reload_configurations()
        
        # Verify failure handling
        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "Config error" in str(result["errors"])
        
        # Verify flows were not changed (rollback)
        assert flow_runner.flows == original_flows

    def test_config_snapshot_handles_missing_attributes(self):
        """Test config snapshot gracefully handles missing run request attributes."""
        flow_runner = FlowRunner(
            flows={"test_flow": {"name": "test_flow"}},
            mode="api"
        )
        
        # Create minimal run request
        run_request = RunRequest(flow="test_flow")
        
        # Should not raise exception
        flow_runner._save_config_snapshot(run_request)
        
        # Verify snapshot was still created with defaults
        expected_session_dir = Path("/tmp/runs/default")
        config_snapshot_dir = expected_session_dir / "config_snapshot"
        
        # Should create directory even with minimal request
        assert config_snapshot_dir.exists()


@pytest.mark.integration
class TestConfigReloadIntegration:
    """Integration tests requiring actual file system operations."""

    def test_startup_script_functionality(self):
        """Test that startup script can be executed without errors."""
        # This would test the actual startup.sh script
        # For now, just verify it exists and is executable
        startup_script = Path("/src/buttermilk/deploy/startup.sh")
        
        # Integration test must fail if startup script not present
        assert startup_script.exists(), "Startup script must be present for integration tests"
        
        # Verify script is executable
        assert startup_script.stat().st_mode & 0o111, "Startup script should be executable"
        
        # Could add more sophisticated testing of script logic here
        # but would require mocking gcsfuse and other container-specific tools

    @pytest.mark.skipif(
        not Path("/usr/bin/gcsfuse").exists(),
        reason="gcsfuse not installed - not in container environment"
    )
    def test_gcsfuse_available(self):
        """Test that gcsfuse is available in the container."""
        import subprocess
        
        # Test gcsfuse help command
        result = subprocess.run(["gcsfuse", "--help"], check=False, capture_output=True, text=True)
        assert result.returncode == 0, "gcsfuse should be available and working"
        assert "gcsfuse" in result.stdout.lower(), "gcsfuse help should mention gcsfuse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
