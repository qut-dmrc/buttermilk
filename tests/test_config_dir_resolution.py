"""Unit tests for config directory resolution logic.

Tests the resolve_config_dir() function that determines where to find
configuration files based on the provided config_dir argument.
"""

import os
from pathlib import Path

import pytest

from buttermilk._core.config_bootstrap import resolve_config_dir


def test_none_finds_cwd_buttermilk_conf(tmp_path, monkeypatch):
    """When config_dir=None and cwd/buttermilk/conf exists, use it."""
    # Arrange: Create cwd/buttermilk/conf
    project_dir = tmp_path / "myproject"
    bm_conf = project_dir / "buttermilk" / "conf"
    bm_conf.mkdir(parents=True)
    monkeypatch.chdir(project_dir)

    # Act
    result = resolve_config_dir(config_dir=None)

    # Assert
    assert result == str(bm_conf.resolve())


def test_none_falls_back_to_packaged_conf(tmp_path, monkeypatch):
    """When config_dir=None and cwd/buttermilk/conf missing, use package conf."""
    # Arrange: Set CWD to directory without buttermilk/conf
    project_dir = tmp_path / "empty_project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    # Act
    result = resolve_config_dir(config_dir=None)

    # Assert: Should return packaged config path
    # The packaged path is relative to config_bootstrap.py
    expected = Path(__file__).parent.parent / "buttermilk" / "conf"
    assert result == str(expected.resolve())


def test_relative_path_resolves_against_cwd(tmp_path, monkeypatch):
    """When config_dir='myconf', resolve to cwd/myconf."""
    # Arrange
    project_dir = tmp_path / "project"
    conf_dir = project_dir / "myconf"
    conf_dir.mkdir(parents=True)
    monkeypatch.chdir(project_dir)

    # Act
    result = resolve_config_dir(config_dir="myconf")

    # Assert
    assert result == str(conf_dir.resolve())


def test_relative_nested_path_resolves_against_cwd(tmp_path, monkeypatch):
    """When config_dir='config/hydra', resolve to cwd/config/hydra."""
    # Arrange
    project_dir = tmp_path / "project"
    conf_dir = project_dir / "config" / "hydra"
    conf_dir.mkdir(parents=True)
    monkeypatch.chdir(project_dir)

    # Act
    result = resolve_config_dir(config_dir="config/hydra")

    # Assert
    assert result == str(conf_dir.resolve())


def test_absolute_path_used_directly(tmp_path):
    """When config_dir='/abs/path', use it directly."""
    # Arrange
    abs_conf = tmp_path / "absolute" / "config"
    abs_conf.mkdir(parents=True)

    # Act
    result = resolve_config_dir(config_dir=str(abs_conf))

    # Assert
    assert result == str(abs_conf.resolve())


def test_tilde_expansion(tmp_path, monkeypatch):
    """When config_dir='~/conf', expand ~ to HOME."""
    # Arrange
    fake_home = tmp_path / "home" / "user"
    fake_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    home_conf = fake_home / "conf"
    home_conf.mkdir()

    # Act
    result = resolve_config_dir(config_dir="~/conf")

    # Assert
    assert result == str(home_conf.resolve())


def test_env_var_expansion(tmp_path, monkeypatch):
    """When config_dir='$MYCONF', expand $MYCONF."""
    # Arrange
    conf_dir = tmp_path / "myconfig"
    conf_dir.mkdir()
    monkeypatch.setenv("MYCONF", str(conf_dir))

    # Act
    result = resolve_config_dir(config_dir="$MYCONF")

    # Assert
    assert result == str(conf_dir.resolve())


def test_env_var_and_tilde_expansion_combined(tmp_path, monkeypatch):
    """When config_dir='$HOME/myconf', expand both."""
    # Arrange
    fake_home = tmp_path / "home" / "user"
    fake_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    home_conf = fake_home / "myconf"
    home_conf.mkdir()

    # Act
    result = resolve_config_dir(config_dir="$HOME/myconf")

    # Assert
    assert result == str(home_conf.resolve())


def test_relative_path_with_tilde_expands_first(tmp_path, monkeypatch):
    """When config_dir='~/relative/path', expand ~ then resolve."""
    # Arrange
    fake_home = tmp_path / "home" / "user"
    fake_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    conf_path = fake_home / "relative" / "path"
    conf_path.mkdir(parents=True)

    # Act
    result = resolve_config_dir(config_dir="~/relative/path")

    # Assert
    assert result == str(conf_path.resolve())
