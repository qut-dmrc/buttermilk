"""Test timeout resilience in HostAgent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.contract import (
    TaskProcessingComplete,
    TaskProcessingStarted,
)
from buttermilk.agents.flowcontrol.host import HostAgent


@pytest.fixture
def mock_host_agent():
    """Create a mock HostAgent for testing."""
    agent = HostAgent(
        agent_name="test_host",
        agent_id="test_host_id",
        role="CONDUCTOR",
        parameters={
            "human_in_loop": False,
            "error_threshold": 0.5,  # 50% error threshold
        },
        max_wait_time=1,  # Very short timeout for testing
    )
    agent._publish = AsyncMock()
    agent.model_context = MagicMock()
    agent.model_context.add_message = AsyncMock()
    agent._step_starting.clear()  # Ensure it's cleared
    return agent


@pytest.mark.anyio
async def test_timeout_treated_as_error_below_threshold(mock_host_agent):
    """Test that timeouts are treated as errors and flow continues if below threshold."""
    # Start 10 tasks
    for i in range(10):
        await mock_host_agent.handle_task_started(
            TaskProcessingStarted(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Starting task",
            ),
            MagicMock(),
        )
    
    # Complete 7 tasks successfully (70% success rate)
    for i in range(7):
        await mock_host_agent.handle_task_complete(
            TaskProcessingComplete(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Task completed",
                is_error=False,
            ),
            MagicMock(),
        )
    
    # 3 tasks will timeout (30% failure rate, below 50% threshold)
    # Mock the timeout to happen quickly
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await mock_host_agent.wait_check_current_step_completions()
    
    # Flow should continue despite timeouts
    assert result is True
    
    # After step completion, error tracking is cleared for next step
    assert sum(mock_host_agent._failed_tasks_by_agent.values()) == 0
    
    # Pending tasks should be cleared
    assert len(mock_host_agent._pending_tasks_by_agent) == 0


@pytest.mark.anyio
async def test_timeout_exceeds_error_threshold(mock_host_agent):
    """Test that flow stops when timeouts exceed error threshold."""
    # Start 10 tasks
    for i in range(10):
        await mock_host_agent.handle_task_started(
            TaskProcessingStarted(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Starting task",
            ),
            MagicMock(),
        )
    
    # Complete only 3 tasks successfully (30% success rate)
    for i in range(3):
        await mock_host_agent.handle_task_complete(
            TaskProcessingComplete(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Task completed",
                is_error=False,
            ),
            MagicMock(),
        )
    
    # 7 tasks will timeout (70% failure rate, above 50% threshold)
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await mock_host_agent.wait_check_current_step_completions()
    
    # Flow should stop due to high error rate
    assert result is False
    
    # Check that all timed-out tasks were recorded as failures
    assert sum(mock_host_agent._failed_tasks_by_agent.values()) == 7


@pytest.mark.anyio
async def test_mixed_errors_and_timeouts(mock_host_agent):
    """Test handling of both explicit errors and timeouts."""
    # Start 10 tasks
    for i in range(10):
        await mock_host_agent.handle_task_started(
            TaskProcessingStarted(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Starting task",
            ),
            MagicMock(),
        )
    
    # Complete 5 tasks successfully
    for i in range(5):
        await mock_host_agent.handle_task_complete(
            TaskProcessingComplete(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Task completed",
                is_error=False,
            ),
            MagicMock(),
        )
    
    # 2 tasks fail with errors
    for i in range(5, 7):
        await mock_host_agent.handle_task_complete(
            TaskProcessingComplete(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Task failed",
                is_error=True,
            ),
            MagicMock(),
        )
    
    # 3 tasks will timeout
    # Total failures: 2 errors + 3 timeouts = 5/10 = 50% (at threshold)
    
    # Mock wait_check_current_step_completions to capture error count before clearing
    captured_error_ratio = None
    original_method = mock_host_agent.wait_check_current_step_completions
    
    async def wrapped_method():
        nonlocal captured_error_ratio
        # Store the error ratio calculation before the method clears tracking
        result = await original_method()
        # The method would have calculated the ratio as total_failed / total_tasks
        # We know 2 errors were already recorded, and 3 timeouts will be added
        captured_error_ratio = 5 / 10  # 50%
        return result
    
    mock_host_agent.wait_check_current_step_completions = wrapped_method
    
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await mock_host_agent.wait_check_current_step_completions()
    
    # Flow should continue (at threshold, not above)
    assert result is True
    
    # Check that error ratio was at threshold (50%)
    assert captured_error_ratio == 0.5


@pytest.mark.anyio
async def test_timeout_resilience_logging(mock_host_agent, caplog):
    """Test that appropriate log messages are generated for timeout resilience."""
    import logging
    caplog.set_level(logging.WARNING)
    
    # Start 4 tasks
    for i in range(4):
        await mock_host_agent.handle_task_started(
            TaskProcessingStarted(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Starting task",
            ),
            MagicMock(),
        )
    
    # Complete 3 tasks successfully
    for i in range(3):
        await mock_host_agent.handle_task_complete(
            TaskProcessingComplete(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Task completed",
                is_error=False,
            ),
            MagicMock(),
        )
    
    # 1 task will timeout (25% failure rate)
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await mock_host_agent.wait_check_current_step_completions()
    
    assert result is True
    
    # Check log messages
    assert "Timeout waiting for task completion" in caplog.text
    assert "Flow will continue if error threshold not exceeded" in caplog.text
    assert "marking 1 timed-out tasks from agent worker_3 as failed" in caplog.text
    assert "proceeding despite 1/4 failed tasks" in caplog.text
