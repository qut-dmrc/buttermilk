"""Integration tests for config_dir resolution in init() function.

These tests verify that the init() function correctly resolves config_dir paths
in various scenarios, ensuring relative paths work as expected.
"""

from pathlib import Path

import pytest

from buttermilk._core.config_bootstrap import resolve_config_dir


class TestInitConfigDirResolution:
    """Test that init() correctly resolves config_dir parameter."""

<<<<<<< HEAD
    def test_resolve_config_dir_with_relative_path_from_cwd(self, tmp_path, monkeypatch):
=======
    def test_resolve_config_dir_with_relative_path_from_cwd(
        self, tmp_path, monkeypatch
    ):
>>>>>>> origin/stable
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

    def test_resolve_config_dir_returns_absolute_path(self, tmp_path, monkeypatch):
        """Test that resolve_config_dir always returns an absolute path.

        Note: We no longer test init_async with custom project names because
        the ExecutionContext enforces a single project name per process.
        The resolve_config_dir function is the key unit to test for path resolution.
        """
        # Arrange: Create a test project structure
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        conf_dir = project_dir / "conf"
        conf_dir.mkdir()

        # Change to project directory
        monkeypatch.chdir(project_dir)

        # Act: Resolve relative path
        result = resolve_config_dir("conf")

        # Assert: Should return an absolute path
        assert Path(result).is_absolute()
        assert result == str(conf_dir.resolve())

<<<<<<< HEAD
    def test_resolve_config_dir_does_not_use_package_dir_for_relative_paths(self, tmp_path, monkeypatch):
=======
    def test_resolve_config_dir_does_not_use_package_dir_for_relative_paths(
        self, tmp_path, monkeypatch
    ):
>>>>>>> origin/stable
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
<<<<<<< HEAD
        assert result != str(package_dir.resolve()), f"config_dir should not resolve to package directory {package_dir}"
=======
        assert result != str(package_dir.resolve()), (
            f"config_dir should not resolve to package directory {package_dir}"
        )
>>>>>>> origin/stable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
