"""Test dynamic timeout calculation in HostAgent."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.contract import (
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
        parameters={"human_in_loop": False},
        max_wait_time=120,  # Base timeout of 2 minutes
    )
    agent._publish = AsyncMock()
    agent._model_context = MagicMock()
    agent._model_context.add_message = AsyncMock()
    return agent


@pytest.mark.anyio
async def test_dynamic_timeout_calculation_in_wait_method(mock_host_agent, caplog):
    """Test that the timeout is calculated correctly based on pending tasks."""
    # Enable debug logging to capture the timeout message
    caplog.set_level(logging.INFO)
    
    # Add multiple pending tasks
    for i in range(6):
        await mock_host_agent.handle_task_started(
            TaskProcessingStarted(
                agent_id=f"worker_{i}",
                role="WORKER",
                content="Starting task",
            ),
            MagicMock(),
        )
    
    # Verify we have the expected number of pending tasks
    async with mock_host_agent._tasks_condition:
        total_pending = sum(mock_host_agent._pending_tasks_by_agent.values())
        assert total_pending == 6
    
    # Patch asyncio.wait_for to capture the timeout value
    captured_timeout = None
    
    async def mock_wait_for(coro, timeout):
        nonlocal captured_timeout
        captured_timeout = timeout
        # Complete immediately to avoid actual waiting
    
    # Clear the starting flag to make the condition satisfiable
    mock_host_agent._step_starting.clear()
    
    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        # Call the method WITHOUT clearing pending tasks first
        result = await mock_host_agent._wait_for_all_tasks_complete()
        
        # Verify the timeout was calculated correctly
        # Expected: max(300, min(120 + (6 * 60 / 6), 1200)) = max(300, min(180, 1200)) = 300
        assert captured_timeout == 300  # Minimum timeout of 5 minutes
        assert result is True
        
        # Verify the log message
        assert "Using dynamic timeout of 300s for 6 pending tasks" in caplog.text


@pytest.mark.anyio
async def test_dynamic_timeout_with_zero_tasks(mock_host_agent, caplog):
    """Test that base timeout is used when no tasks are pending."""
    caplog.set_level(logging.INFO)
    
    # No tasks added, so pending should be 0
    async with mock_host_agent._tasks_condition:
        assert len(mock_host_agent._pending_tasks_by_agent) == 0
    
    captured_timeout = None
    
    async def mock_wait_for(coro, timeout):
        nonlocal captured_timeout
        captured_timeout = timeout
    
    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        mock_host_agent._step_starting.clear()
        
        result = await mock_host_agent._wait_for_all_tasks_complete()
        
        # Should use minimum timeout of 5 minutes (300s)
        assert captured_timeout == 300
        assert result is True
        
        # Verify the log message
        assert "Using dynamic timeout of 300s for 0 pending tasks" in caplog.text


@pytest.mark.anyio
async def test_dynamic_timeout_with_varying_task_counts(mock_host_agent):
    """Test timeout calculation with different numbers of tasks."""
    test_cases = [
        (1, 300),     # max(300, min(120 + (1 * 60 / 6), 1200)) = max(300, min(130, 1200)) = 300 (min limit)
        (3, 300),     # max(300, min(120 + (3 * 60 / 6), 1200)) = max(300, min(150, 1200)) = 300 (min limit)
        (10, 300),    # max(300, min(120 + (10 * 60 / 6), 1200)) = max(300, min(220, 1200)) = 300 (min limit)
        (18, 300),    # max(300, min(120 + (18 * 60 / 6), 1200)) = max(300, min(300, 1200)) = 300
        (30, 420),    # max(300, min(120 + (30 * 60 / 6), 1200)) = max(300, min(420, 1200)) = 420
        (60, 720),    # max(300, min(120 + (60 * 60 / 6), 1200)) = max(300, min(720, 1200)) = 720
        (120, 1200),  # max(300, min(120 + (120 * 60 / 6), 1200)) = max(300, min(1320, 1200)) = 1200 (max limit)
    ]
    
    for num_tasks, expected_timeout in test_cases:
        # Reset state
        mock_host_agent._pending_tasks_by_agent.clear()
        
        # Add tasks
        for i in range(num_tasks):
            await mock_host_agent.handle_task_started(
                TaskProcessingStarted(
                    agent_id=f"worker_{i}",
                    role="WORKER",
                    content="Starting task",
                ),
                MagicMock(),
            )
        
        captured_timeout = None
        
        async def mock_wait_for(coro, timeout):
            nonlocal captured_timeout
            captured_timeout = timeout
        
        # Clear the starting flag before calling the method
        mock_host_agent._step_starting.clear()
        
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            # Call WITHOUT clearing pending tasks to test timeout calculation
            await mock_host_agent._wait_for_all_tasks_complete()
            
            assert captured_timeout == expected_timeout, f"Failed for {num_tasks} tasks. Got {captured_timeout}, expected {expected_timeout}"


@pytest.mark.anyio
async def test_dynamic_timeout_respects_limits(mock_host_agent):
    """Test that timeout respects min/max limits."""
    # Test minimum limit
    mock_host_agent._max_wait_time = 10  # Very low base timeout
    
    # Add just 1 task
    await mock_host_agent.handle_task_started(
        TaskProcessingStarted(
            agent_id="worker_1",
            role="WORKER",
        ),
        MagicMock(),
    )
    
    captured_timeout = None
    
    async def mock_wait_for(coro, timeout):
        nonlocal captured_timeout
        captured_timeout = timeout
    
    mock_host_agent._step_starting.clear()
    
    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        await mock_host_agent._wait_for_all_tasks_complete()
        
        # Should still use minimum of 300s even with low base timeout
        assert captured_timeout == 300
    
    # Test maximum limit
    mock_host_agent._pending_tasks_by_agent.clear()
    mock_host_agent._max_wait_time = 1000  # High base timeout
    
    # Add many tasks
    for i in range(200):
        await mock_host_agent.handle_task_started(
            TaskProcessingStarted(
                agent_id=f"worker_{i}",
                role="WORKER",
            ),
            MagicMock(),
        )
    
    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        await mock_host_agent._wait_for_all_tasks_complete()
        
        # Should be capped at 1200s (20 minutes)
        assert captured_timeout == 1200
