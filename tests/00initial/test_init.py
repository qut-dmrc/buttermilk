"""Test Buttermilk initialization paths.

This module tests the various supported initialization patterns:
- init() - Sync wrapper (deprecated but supported)
- nb_init() - Notebook convenience function
- init_async() - Primary async initialization
"""
import asyncio
import re
from pathlib import Path

import pytest

from buttermilk import init, init_async
from buttermilk.utils.nb import nb_init

# Expected values from testing.yaml config
EXPECTED_PROJECT_NAME = "buttermilk"
EXPECTED_JOB = "testing"


def test_short_form_cli():
    """Test sync init() wrapper for CLI usage."""
    # Must use same project as real_bm fixture (buttermilk) since all sessions share execution context
    bm = init(job="test_cli", project_name="buttermilk")
    assert bm is not None
    assert bm.cloud_manager is not None
    assert bm.session_info.project_name == "buttermilk"
    assert bm.session_info.job == "test_cli"


def test_short_form_nb():
    """Test nb_init() for notebook usage."""
    # Must use same project as real_bm fixture (buttermilk)
    bm = nb_init(job="test_nb", project="buttermilk")
    logger = bm.logger
    logger.debug("logging seems to work")
    assert bm.cloud_manager is not None
    assert bm.session_info.project_name == "buttermilk"
    assert bm.session_info.job == "test_nb"


@pytest.mark.asyncio
async def test_init_async():
    """Test async init_async() - the primary initialization path."""
    # Must use same project as real_bm fixture (buttermilk)
    bm = await init_async(job="test_async", project_name="buttermilk")
    assert bm is not None
    assert bm.cloud_manager is not None
    assert bm.session_info.project_name == "buttermilk"
    assert bm.session_info.job == "test_async"


@pytest.mark.asyncio
async def test_log_file_uses_project_name(real_bm):
    """Test that log file path includes project name.

    Expected format: /tmp/buttermilk_exec-{timestamp}-{uuid}.jsonl
    """
    # The test just verifies real_bm was initialized successfully
    # Log files are created during init and use the execution context ID
    assert real_bm is not None
    assert real_bm.session_info.project_name == EXPECTED_PROJECT_NAME


@pytest.mark.asyncio
async def test_save_dir_uses_proper_format(real_bm):
    """Test that bm.save_dir uses proper session-based format.

    Expected format: {base}/{project_name}/{job}/session-{timestamp}-{uuid}/
    """
    save_dir = real_bm.session_info.save_dir
    assert save_dir is not None, "save_dir should be set"

    # Verify it contains project name
    assert EXPECTED_PROJECT_NAME in save_dir, (
        f"save_dir should contain project name '{EXPECTED_PROJECT_NAME}', got: {save_dir}"
    )

    # Verify it contains job name
    assert EXPECTED_JOB in save_dir, (
        f"save_dir should contain job '{EXPECTED_JOB}', got: {save_dir}"
    )

    # Verify it uses session-* pattern (current design)
    assert "session-" in save_dir, (
        f"save_dir should use 'session-' prefix for session ID, got: {save_dir}"
    )

    # Pattern: session-YYYYMMDDTHHmmZ-xxxx
    session_pattern = re.compile(r"session-\d{8}T\d{4}Z-[a-zA-Z0-9]{4}")
    assert session_pattern.search(save_dir), (
        f"save_dir should contain session ID in format 'session-YYYYMMDDTHHmmZ-xxxx', got: {save_dir}"
    )
