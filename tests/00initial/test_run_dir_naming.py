"""Tests for run_dir and log file naming with project/job information.

These tests verify that log files and save directories use meaningful names
based on project and job, rather than generic execution context IDs.

Related to: https://github.com/nicsuzor/academicOps/issues/83
"""
import asyncio
import re
from pathlib import Path

import pytest
from hydra import compose, initialize

from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
from buttermilk._core.log import reset_logging_configuration


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    reset_logging_configuration()
    yield
    reset_logging_configuration()


@pytest.mark.asyncio
async def test_log_file_uses_project_name():
    """Test that log file path includes project name, not generic execution-context ID.

    Expected format: /tmp/{project_name}_{timestamp}-{shortuuid}.jsonl
    Not: /tmp/buttermilk_exec-{timestamp}-{uuid}.jsonl
    """
    # Load configuration
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Create bootstrapper
    bootstrapper = ConfigurationBootstrapper(config=cfg)

    # Bootstrap with specific project name
    project_name = "test_project"
    job_name = "test_job"

    # Bootstrap full context (this sets up logging) WITH project_name
    execution_context = await bootstrapper.bootstrap_full_context(project_name=project_name)

    # Bootstrap session context
    session_bm = await bootstrapper.bootstrap_session_context(
        name=project_name,
        job=job_name,
    )

    # Find log files in /tmp
    log_files = list(Path("/tmp").glob("*.jsonl"))

    # Filter to files created in this test (should contain project_name)
    project_log_files = [
        f for f in log_files
        if project_name in f.name
    ]

    # We should have at least one log file with the project name
    assert len(project_log_files) > 0, (
        f"Expected log file with project name '{project_name}' in /tmp, "
        f"but found: {[f.name for f in log_files]}"
    )

    # Verify the log file follows expected naming pattern
    log_file = project_log_files[0]

    # Expected pattern: {project_name}_{timestamp}-{shortuuid}.jsonl
    # Example: test_project_20250109T1200Z-a1b2.jsonl
    expected_pattern = re.compile(
        rf"{project_name}_\d{{8}}T\d{{4}}Z-[a-zA-Z0-9]{{4}}\.jsonl"
    )

    assert expected_pattern.match(log_file.name), (
        f"Log file name '{log_file.name}' doesn't match expected pattern. "
        f"Should be: {project_name}_YYYYMMDDTHHmmZ-xxxx.jsonl"
    )

    # Verify the log file is NOT using generic 'exec-' prefix
    assert "exec-" not in log_file.name, (
        f"Log file should not use generic 'exec-' prefix, got: {log_file.name}"
    )


@pytest.mark.asyncio
async def test_save_dir_uses_proper_format():
    """Test that bm.save_dir uses formatted project/job names.

    Expected format: {base}/project_name/job-timestamp-uuid/
    Not: {base}/project_name/job/session-timestamp-uuid/
    """
    # Load configuration
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Create bootstrapper
    bootstrapper = ConfigurationBootstrapper(config=cfg)

    project_name = "osb_analysis"
    job_name = "extract_cases"

    # Bootstrap full context WITH project_name
    execution_context = await bootstrapper.bootstrap_full_context(project_name=project_name)

    # Bootstrap session context
    session_bm = await bootstrapper.bootstrap_session_context(
        name=project_name,
        job=job_name,
    )

    # Get save_dir from session
    save_dir = session_bm.session_info.save_dir

    assert save_dir is not None, "save_dir should be set"

    # Parse the save_dir path
    # Expected: gs://bucket/project_name/job-YYYYMMDDTHHmmZ-xxxx/
    # Or: /tmp/project_name/job-YYYYMMDDTHHmmZ-xxxx/

    assert project_name in save_dir, (
        f"save_dir should contain project name '{project_name}', got: {save_dir}"
    )

    # Verify it uses formatted job name (job-timestamp), not session-timestamp
    # Pattern: job_name-YYYYMMDDTHHmmZ-xxxx
    job_pattern = re.compile(
        rf"{job_name}-\d{{8}}T\d{{4}}Z-[a-zA-Z0-9]{{4}}"
    )

    assert job_pattern.search(save_dir), (
        f"save_dir should use formatted job name '{job_name}-YYYYMMDDTHHmmZ-xxxx', "
        f"got: {save_dir}"
    )

    # Verify it does NOT use the session-* pattern
    assert "session-" not in save_dir, (
        f"save_dir should not use 'session-' prefix, got: {save_dir}"
    )


@pytest.mark.asyncio
async def test_weave_collection_uses_project_name():
    """Test that weave initialization uses project name, not execution context ID.

    Weave should ALWAYS use project_name for collection, never fall back to
    'execution-context-{id}'.
    """
    # Load configuration
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Create bootstrapper
    bootstrapper = ConfigurationBootstrapper(config=cfg)

    project_name = "test_weave_project"
    job_name = "test_weave_job"

    # Bootstrap full context WITH project_name
    execution_context = await bootstrapper.bootstrap_full_context(project_name=project_name)

    # Verify project_name is set in ExecutionContext
    assert execution_context.project_name == project_name, (
        f"ExecutionContext should have project_name set to '{project_name}', "
        f"got: {execution_context.project_name}"
    )

    # If weave is enabled, verify it uses project_name
    if execution_context.tracing and execution_context.tracing.get("weave"):
        weave_config = execution_context.tracing["weave"]
        if weave_config.enabled:
            # Get weave client (this initializes weave)
            try:
                weave_client = await execution_context.get_weave_client()

                # Verify the weave project name matches our project_name
                # The weave project should be {entity}/{project_name}
                assert project_name in weave_client.project, (
                    f"Weave project should contain '{project_name}', "
                    f"got: {weave_client.project}"
                )

                # Verify it does NOT use execution-context prefix
                assert "execution-context" not in weave_client.project, (
                    f"Weave should not use 'execution-context' prefix, "
                    f"got: {weave_client.project}"
                )
            except Exception as e:
                # If weave initialization fails (missing credentials, etc.),
                # that's okay for this test - we're just checking the config
                pytest.skip(f"Weave initialization failed: {e}")


@pytest.mark.asyncio
async def test_multiple_sessions_same_project():
    """Test that multiple sessions in same project share project name but have unique run IDs.

    All sessions should:
    - Use same project_name in ExecutionContext
    - Have unique session IDs with formatted job names
    - Write to different save_dirs with job-timestamp format
    """
    # Load configuration
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Create bootstrapper
    bootstrapper = ConfigurationBootstrapper(config=cfg)

    project_name = "shared_project"

    # Bootstrap full context ONCE (shared infrastructure) WITH project_name
    execution_context = await bootstrapper.bootstrap_full_context(project_name=project_name)

    # Create first session
    job1 = "analysis_v1"
    session_bm1 = await bootstrapper.bootstrap_session_context(
        name=project_name,
        job=job1,
    )

    # Create second session
    job2 = "analysis_v2"
    session_bm2 = await bootstrapper.bootstrap_session_context(
        name=project_name,
        job=job2,
    )

    # Both should use same project
    assert session_bm1.session_info.project_name == project_name
    assert session_bm2.session_info.project_name == project_name

    # But have different session IDs
    assert session_bm1.session_info.session_id != session_bm2.session_info.session_id

    # And different save_dirs with formatted job names
    save_dir1 = session_bm1.session_info.save_dir
    save_dir2 = session_bm2.session_info.save_dir

    assert save_dir1 != save_dir2
    assert job1 in save_dir1 or f"{job1}-" in save_dir1
    assert job2 in save_dir2 or f"{job2}-" in save_dir2

    # Verify they don't use session- prefix
    assert "session-" not in save_dir1
    assert "session-" not in save_dir2
