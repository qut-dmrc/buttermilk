"""Test BM async initialization behavior.

This test verifies that:
1. BM instance initializes properly with session info
2. Async initialization completes correctly
3. Session logging context is established
4. Save directory is properly constructed
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from buttermilk._core.bm_init import BM, SessionInfo, create_session_bm_async


class TestBMAsyncInitialization:
    """Test BM async initialization behavior."""

    @pytest.mark.anyio
    async def test_initialization_completes_before_use(self):
        """Test that ensure_initialized waits for background tasks."""
        # Create SessionInfo
        session_info = SessionInfo(
            project_name="test",
            job="test-job",
        )

        # Create BM instance with minimal setup
        bm = BM(session_info=session_info, save_dir_base="/tmp/test-runs")

        # Perform async initialization
        await bm._async_init()

        # Verify initialization event is created
        assert hasattr(bm, "_initialization_complete")
        assert hasattr(bm, "_initialization_error")

        # Wait for initialization
        await bm.ensure_initialized()

        # Verify initialization is complete
        assert bm._initialization_complete.is_set()
        assert bm._initialization_error is None

        # Verify save_dir is properly set
        assert bm.session_info.save_dir.startswith("/tmp/test-runs")
        assert "test/test-job" in bm.session_info.save_dir

    @pytest.mark.anyio
    async def test_initialization_error_handling(self):
        """Test that initialization errors are properly propagated."""
        session_info = SessionInfo(
            project_name="test",
            job="test-job",
        )

        bm = BM(session_info=session_info)

        # Simulate an error during async init by patching a method
        with patch.object(bm, "_finalize_save_dir", side_effect=Exception("Test error")):
            await bm._async_init()

            # Should raise error when waiting for initialization
            with pytest.raises(RuntimeError, match="Session initialization failed"):
                await bm.ensure_initialized()

    @pytest.mark.anyio
    async def test_multiple_ensure_initialized_calls(self):
        """Test that ensure_initialized can be called multiple times safely."""
        session_info = SessionInfo(
            project_name="test",
            job="test-job",
        )

        bm = BM(session_info=session_info)
        await bm._async_init()

        # Call ensure_initialized multiple times
        await bm.ensure_initialized()
        await bm.ensure_initialized()
        await bm.ensure_initialized()

        # Should not raise any errors
        assert bm._initialization_complete.is_set()

    @pytest.mark.anyio
    async def test_create_session_bm_async_factory(self):
        """Test the async factory function for creating BM instances."""
        # Create mock managers
        mock_cloud_manager = Mock()
        mock_secret_manager = Mock()
        mock_llms = Mock()
        mock_query_runner = Mock()  # Provide query_runner to avoid auto-creation

        bm = await create_session_bm_async(
            project_name="test-project",
            job="test-job",
            platform="test",
            cloud_manager=mock_cloud_manager,
            secret_manager=mock_secret_manager,
            llms_instance=mock_llms,
            query_runner=mock_query_runner,  # Provide explicitly to avoid validation error
        )

        # Verify BM was created and initialized
        assert bm.session_info.project_name == "test-project"
        assert bm.session_info.job == "test-job"
        assert bm.session_info.platform == "test"
        assert bm._initialization_complete.is_set()

        # Verify injected dependencies are accessible
        assert bm._cloud_manager == mock_cloud_manager
        assert bm._secret_manager == mock_secret_manager
        assert bm._llms_instance == mock_llms
        assert bm._query_runner == mock_query_runner

    @pytest.mark.anyio
    async def test_gcs_save_dir_construction(self):
        """Test that GCS save_dir paths are properly constructed."""
        session_info = SessionInfo(
            project_name="test",
            job="test-job",
        )

        bm = BM(session_info=session_info, save_dir_base="gs://test-bucket/runs")
        await bm._async_init()

        # Verify save_dir includes GCS path
        assert bm.session_info.save_dir.startswith("gs://test-bucket/runs")
        assert "test/test-job" in bm.session_info.save_dir

    @pytest.mark.anyio
    async def test_config_storage(self):
        """Test that config is properly stored on BM instance."""
        from omegaconf import DictConfig

        session_info = SessionInfo(
            project_name="test",
            job="test-job",
        )

        test_config = DictConfig({"test_key": "test_value"})

        bm = await create_session_bm_async(
            project_name="test",
            job="test-job",
            config=test_config,
        )

        # Config should be auto-instantiated and stored
        assert bm._config is not None
        # After instantiation, config is converted to plain dict/object
        assert bm.cfg["test_key"] == "test_value"

    @pytest.mark.anyio
    async def test_session_info_creation(self):
        """Test SessionInfo creation and auto-generated fields."""
        session_info = SessionInfo(
            project_name="test-project",
            job="test-job",
        )

        # Verify auto-generated session_id
        assert session_info.session_id is not None
        assert session_info.session_id.startswith("session-")

        # Verify default values
        assert session_info.platform == "local"
        assert session_info.status == "initializing"
        assert session_info.records_processed == 0
        assert session_info.outputs_generated == 0

    @pytest.mark.anyio
    async def test_cloud_manager_property_error_when_not_injected(self):
        """Test that accessing cloud_manager raises error when not injected."""
        session_info = SessionInfo(
            project_name="test",
            job="test-job",
        )

        bm = BM(session_info=session_info)

        # Should raise error when accessing cloud_manager without injection
        with pytest.raises(RuntimeError, match="CloudManager not available"):
            _ = bm.cloud_manager

    @pytest.mark.anyio
    async def test_secret_manager_property_error_when_not_injected(self):
        """Test that accessing secret_manager raises error when not injected."""
        session_info = SessionInfo(
            project_name="test",
            job="test-job",
        )

        bm = BM(session_info=session_info)

        # Should raise error when accessing secret_manager without injection
        with pytest.raises(RuntimeError, match="SecretsManager not available"):
            _ = bm.secret_manager

    @pytest.mark.anyio
    async def test_save_initial_config_saves_full_cfg(self, tmp_path):
        """Test that _save_initial_config saves the entire .cfg object."""
        import json
        from pathlib import Path
        from omegaconf import DictConfig

        # Create a test config with nested structure
        test_config = DictConfig({
            "project_name": "test-project",
            "job": "test-job",
            "nested": {
                "key1": "value1",
                "key2": 42,
                "deep": {
                    "key3": "value3"
                }
            },
            "list_field": [1, 2, 3]
        })

        # Create BM instance with config
        bm = await create_session_bm_async(
            project_name="test-project",
            job="test-job",
            save_dir_base=str(tmp_path),
            config=test_config,
        )

        # Find the saved config file (currently saved as tmp*.json due to dump_to_disk behavior)
        save_dir = Path(bm.session_info.save_dir)
        config_files = list(save_dir.glob("*.json"))
        assert len(config_files) == 1, f"Should have exactly one JSON file, found: {config_files}"

        # Read the saved config
        with open(config_files[0]) as f:
            saved_data = json.load(f)

        # Debug: print what's actually saved
        import pprint
        print("\n=== Saved data structure ===")
        pprint.pprint(saved_data, depth=2)
        print("\n=== Keys in saved_data ===")
        print(list(saved_data.keys()))
        if "cfg" in saved_data and saved_data["cfg"]:
            print("\n=== Keys in cfg ===")
            print(list(saved_data["cfg"].keys()))

        # Verify the saved data contains the full config
        assert isinstance(saved_data, dict), f"Saved data should be a dict, got: {type(saved_data)}"
        assert "cfg" in saved_data, "Saved data should contain 'cfg' key"
        assert saved_data["cfg"]["project_name"] == "test-project"
        assert saved_data["cfg"]["nested"]["key1"] == "value1"
        assert saved_data["cfg"]["nested"]["deep"]["key3"] == "value3"
        assert saved_data["cfg"]["list_field"] == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
