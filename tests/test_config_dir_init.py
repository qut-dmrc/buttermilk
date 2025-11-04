"""Integration tests for config_dir resolution in init() function.

These tests verify that the init() function correctly resolves config_dir paths
in various scenarios, ensuring relative paths work as expected.
"""

from pathlib import Path

import pytest

from buttermilk._core.config_bootstrap import resolve_config_dir


class TestInitConfigDirResolution:
    """Test that init() correctly resolves config_dir parameter."""

    def test_resolve_config_dir_with_relative_path_from_cwd(self, tmp_path, monkeypatch):
        """Test that relative config_dir is resolved against CWD, not package dir."""
        # Arrange: Create a test project structure
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        conf_dir = project_dir / "conf"
        conf_dir.mkdir()

        # Change to project directory
        monkeypatch.chdir(project_dir)

        # Act: Resolve relative path "conf"
        result = resolve_config_dir("conf")

        # Assert: Should resolve to CWD/conf, not package/conf
        expected = str(conf_dir.resolve())
        assert result == expected, f"Expected {expected} but got {result}"

    def test_resolve_config_dir_with_nested_relative_path(self, tmp_path, monkeypatch):
        """Test that nested relative paths like '../conf' work correctly."""
        # Arrange: Create a test project structure
        project_dir = tmp_path / "myproject"
        conf_dir = project_dir / "conf"
        conf_dir.mkdir(parents=True)

        subdir = project_dir / "src"
        subdir.mkdir()

        # Change to subdirectory
        monkeypatch.chdir(subdir)

        # Act: Resolve relative path "../conf"
        result = resolve_config_dir("../conf")

        # Assert: Should resolve to parent/conf
        expected = str(conf_dir.resolve())
        assert result == expected, f"Expected {expected} but got {result}"

    def test_resolve_config_dir_with_dot_slash_prefix(self, tmp_path, monkeypatch):
        """Test that './conf' works correctly."""
        # Arrange: Create a test project structure
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        conf_dir = project_dir / "conf"
        conf_dir.mkdir()

        # Change to project directory
        monkeypatch.chdir(project_dir)

        # Act: Resolve relative path "./conf"
        result = resolve_config_dir("./conf")

        # Assert: Should resolve to CWD/conf
        expected = str(conf_dir.resolve())
        assert result == expected, f"Expected {expected} but got {result}"

    @pytest.mark.anyio
    async def test_init_async_with_relative_config_dir(self, tmp_path, monkeypatch):
        """Test that init_async() works with relative config_dir."""

        from buttermilk._core.config_bootstrap import init_async

        # Arrange: Create a minimal config structure
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        conf_dir = project_dir / "conf"
        conf_dir.mkdir()

        # Create a minimal config.yaml
        minimal_config = """
project_name: test_project

bm:
  session_info:
    project_name: ${project_name}
    job: test_job
    cache_dir: ${oc.env:HOME}/.cache/buttermilk
    sessions_dir: ${oc.env:HOME}/.cache/buttermilk/sessions
    session:
      timeout_minutes: 60
      cleanup_interval_minutes: 15
      max_concurrent_sessions: 50
    template_paths: []

infrastructure:
  clouds: []
  logging:
    type: local
    level: INFO
  tracing:
    weave:
      enabled: false
    traceloop:
      enabled: false
    otel:
      enabled: false
  datasets: {}

run:
  mode: console
  human_in_loop: false

flows: {}
storage: {}
pipeline: null
"""
        config_file = conf_dir / "config.yaml"
        config_file.write_text(minimal_config)

        # Change to project directory
        monkeypatch.chdir(project_dir)

        # Act: Initialize with relative config_dir
        try:
            bm = await init_async(config_dir="conf", project_name="test_project", job="test_job")

            # Assert: Should successfully initialize
            assert bm is not None
            assert bm.session_info.project_name == "test_project"
            assert bm.session_info.job == "test_job"
        except Exception as e:
            pytest.fail(f"init_async() failed with relative config_dir: {e}")

    def test_resolve_config_dir_does_not_use_package_dir_for_relative_paths(self, tmp_path, monkeypatch):
        """Test that relative paths are NOT resolved relative to the package directory.

        This is the core bug we're fixing: when a user specifies config_dir="conf",
        it should resolve to CWD/conf, not package_install_dir/conf.
        """
        # Arrange: Create a project directory different from package directory
        project_dir = tmp_path / "user_project"
        project_dir.mkdir()
        conf_dir = project_dir / "conf"
        conf_dir.mkdir()

        # Change to project directory
        monkeypatch.chdir(project_dir)

        # Act: Resolve relative path
        result = resolve_config_dir("conf")

        # Assert: Should resolve to project_dir/conf, not package_dir/conf
        expected = str(conf_dir.resolve())
        assert result == expected

        # Also verify it's not using the package directory
        package_dir = Path(__file__).parent.parent / "buttermilk" / "conf"
        assert result != str(package_dir.resolve()), f"config_dir should not resolve to package directory {package_dir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
