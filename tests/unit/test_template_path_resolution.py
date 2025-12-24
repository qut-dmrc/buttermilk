"""Test template path resolution with config_source_dir parameter.

This test verifies that when config_source_dir is provided to
bootstrap_session_with_config_async(), relative template paths resolve
against config_source_dir rather than config_dir or CWD.
"""

from pathlib import Path

import pytest
import yaml

from buttermilk._core.config_bootstrap import bootstrap_session_with_config_async


@pytest.mark.anyio
async def test_template_paths_resolve_against_config_source_dir(tmp_path: Path):
    """Test that config_source_dir takes precedence for relative template path resolution.

    When config_source_dir is provided, relative template paths (like "../templates")
    should resolve against config_source_dir, NOT against config_dir or CWD.

    This ensures that configs loaded from external sources (like GCS) can reference
    templates relative to their original location, not the local cache directory.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Expected behavior:
        - Relative template path "../templates" resolves to tmp_path/external/templates
        - Resolution uses config_source_dir (tmp_path/external/conf) as base
        - Does NOT resolve against config_dir or current working directory
    """
    # Set up directory structure:
    # tmp_path/
    #   external/
    #     conf/           <- config_source_dir (original source location)
    #       config.yaml
    #     templates/      <- where templates actually are
    #       test.jinja2
    #   cache/
    #     conf/           <- config_dir (local cache)
    #       config.yaml   <- copied here

    config_source_dir = tmp_path / "external" / "conf"
    config_source_dir.mkdir(parents=True)

    templates_dir = tmp_path / "external" / "templates"
    templates_dir.mkdir(parents=True)

    config_cache_dir = tmp_path / "cache" / "conf"
    config_cache_dir.mkdir(parents=True)

    # Create a template file in the external templates directory
    template_file = templates_dir / "test.jinja2"
    template_file.write_text("Hello {{ name }}!")

    # Create config with relative template path
    # "../templates" should resolve relative to config_source_dir
    config_data = {
        "session": {
            "template_paths": ["../templates"],  # Relative path
            "template_env": "jinja2",
        },
        "project": {
            "name": "test_project",
        },
    }

    # Write config to both locations (simulating GCS download to cache)
    config_source_file = config_source_dir / "config.yaml"
    config_cache_file = config_cache_dir / "config.yaml"

    with open(config_source_file, "w") as f:
        yaml.dump(config_data, f)

    with open(config_cache_file, "w") as f:
        yaml.dump(config_data, f)

    # Call bootstrap with BOTH config_dir (cache) and config_source_dir (original)
    # The config_source_dir parameter should take precedence for path resolution
    bm, typed_config = await bootstrap_session_with_config_async(
        job="test_job",
        project_name="test_project",
        config_dir=str(config_cache_dir),  # Local cache location
        config_source_dir=str(config_source_dir),  # Original source location
        config_name="config",
    )

    # Assert: template path should resolve against config_source_dir
    # Expected: tmp_path/external/templates (relative to config_source_dir)
    # NOT: tmp_path/cache/templates (relative to config_dir)
    # NOT: cwd/../templates (relative to current directory)

    assert len(typed_config.session.template_paths) == 1

    resolved_path = Path(typed_config.session.template_paths[0])

    # Should resolve to the external templates directory
    expected_path = templates_dir.resolve()

    assert (
        resolved_path == expected_path
    ), f"Expected {expected_path}, got {resolved_path}"

    # Verify the resolved path actually contains our template
    assert (resolved_path / "test.jinja2").exists(), (
        f"Template file not found at resolved path {resolved_path}"
    )
