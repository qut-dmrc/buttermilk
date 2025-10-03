"""Tests for HostAgent error handling functionality."""

from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from buttermilk._core.contract import TaskProcessingComplete, TaskProcessingStarted
from buttermilk.agents.flowcontrol.host import HostAgent


class TestHostAgentErrorHandling:
    """Test suite for HostAgent error handling functionality."""

    @pytest.fixture
    def host_agent(self):
        """Create a host agent instance for testing."""
        return HostAgent(
            role="HOST",
            description="Test host agent",
            parameters={
                "human_in_loop": False,
                "error_threshold": 0.5,  # 50% error threshold for testing
            },
            unique_identifier="test_host",
        )

    @pytest.fixture
    def mock_message_context(self):
        """Create a mock message context."""
        mock_ctx = MagicMock()
        mock_ctx.sender = None
        return mock_ctx

    @pytest.mark.anyio
    async def test_error_tracking_initialization(self, host_agent):
        """Test that error tracking fields are properly initialized."""
        assert hasattr(host_agent, "_failed_tasks_by_agent")
        assert hasattr(host_agent, "_total_tasks_in_step")
        assert hasattr(host_agent, "_error_threshold")
        
        assert isinstance(host_agent._failed_tasks_by_agent, defaultdict)
        assert host_agent._total_tasks_in_step == 0
        assert host_agent._error_threshold == 0.5

    @pytest.mark.anyio
    async def test_task_started_tracking(self, host_agent, mock_message_context):
        """Test that task start events properly increment counters."""
        # Simulate 3 tasks starting
        for i in range(3):
            start_msg = TaskProcessingStarted(
                agent_id=f"agent_{i}",
                role="WORKER",
                task_index=0
            )
            await host_agent.handle_task_started(start_msg, mock_message_context)
        
        # Check that counters are properly updated
        assert host_agent._total_tasks_in_step == 3
        assert sum(host_agent._pending_tasks_by_agent.values()) == 3

    @pytest.mark.anyio
    async def test_task_completion_error_tracking(self, host_agent, mock_message_context):
        """Test that task completion properly tracks errors."""
        # Start 3 tasks
        for i in range(3):
            start_msg = TaskProcessingStarted(
                agent_id=f"agent_{i}",
                role="WORKER",
                task_index=0
            )
            await host_agent.handle_task_started(start_msg, mock_message_context)
        
        # Complete 2 tasks with errors
        for i in range(2):
            complete_msg = TaskProcessingComplete(
                agent_id=f"agent_{i}",
                role="WORKER",
                task_index=0,
                is_error=True
            )
            await host_agent.handle_task_complete(complete_msg, mock_message_context)
        
        # Complete 1 task successfully
        complete_msg = TaskProcessingComplete(
            agent_id="agent_2",
            role="WORKER",
            task_index=0,
            is_error=False
        )
        await host_agent.handle_task_complete(complete_msg, mock_message_context)
        
        # Check error tracking
        assert sum(host_agent._failed_tasks_by_agent.values()) == 2
        assert host_agent._total_tasks_in_step == 3

    @pytest.mark.anyio
    async def test_wait_check_completions_stops_on_high_errors(self, host_agent, mock_message_context):
        """Test that flow stops when error threshold is exceeded."""
        # Mock the _wait_for_all_tasks_complete method to return True (tasks completed)
        host_agent._wait_for_all_tasks_complete = AsyncMock(return_value=True)
        
        # Simulate scenario: 3 out of 5 tasks failed (60% > 50% threshold)
        host_agent._total_tasks_in_step = 5
        host_agent._failed_tasks_by_agent = defaultdict(int, {"agent_1": 2, "agent_2": 1})
        
        # Should return False (stop flow)
        result = await host_agent.wait_check_current_step_completions()
        assert result is False

    @pytest.mark.anyio
    async def test_wait_check_completions_continues_on_low_errors(self, host_agent, mock_message_context):
        """Test that flow continues when error threshold is not exceeded."""
        # Mock the _wait_for_all_tasks_complete method to return True (tasks completed)
        host_agent._wait_for_all_tasks_complete = AsyncMock(return_value=True)
        
        # Simulate scenario: 2 out of 5 tasks failed (40% < 50% threshold)
        host_agent._total_tasks_in_step = 5
        host_agent._failed_tasks_by_agent = defaultdict(int, {"agent_1": 1, "agent_2": 1})
        
        # Should return True (continue flow)
        result = await host_agent.wait_check_current_step_completions()
        assert result is True

    @pytest.mark.anyio
    async def test_wait_check_completions_clears_tracking(self, host_agent, mock_message_context):
        """Test that error tracking is cleared after successful step completion."""
        # Mock the _wait_for_all_tasks_complete method to return True
        host_agent._wait_for_all_tasks_complete = AsyncMock(return_value=True)
        
        # Set up some error tracking data
        host_agent._total_tasks_in_step = 5
        host_agent._failed_tasks_by_agent = defaultdict(int, {"agent_1": 1})
        host_agent._pending_tasks_by_agent = defaultdict(int, {"agent_1": 0})

        # Call wait_check_current_step_completions
        result = await host_agent.wait_check_current_step_completions()
        assert result is True
        
        # Check that tracking is cleared
        assert host_agent._total_tasks_in_step == 0
        assert len(host_agent._failed_tasks_by_agent) == 0
        assert len(host_agent._pending_tasks_by_agent) == 0

    @pytest.mark.anyio
    async def test_custom_error_threshold(self):
        """Test that custom error thresholds work correctly."""
        # Create host with 25% error threshold
        strict_host = HostAgent(
            role="HOST",
            description="Strict host agent",
            parameters={"human_in_loop": False, "error_threshold": 0.25},
            unique_identifier="strict_host",
        )
        
        # Mock the _wait_for_all_tasks_complete method
        strict_host._wait_for_all_tasks_complete = AsyncMock(return_value=True)
        
        # Simulate scenario: 2 out of 10 tasks failed (20% < 25% threshold)
        strict_host._total_tasks_in_step = 10
        strict_host._failed_tasks_by_agent = defaultdict(int, {"agent_1": 1, "agent_2": 1})
        
        # Should continue (below threshold)
        result = await strict_host.wait_check_current_step_completions()
        assert result is True
        
        # Now test with 3 out of 10 tasks failed (30% > 25% threshold)
        strict_host._total_tasks_in_step = 10
        strict_host._failed_tasks_by_agent = defaultdict(int, {"agent_1": 2, "agent_2": 1})
        
        # Should stop (above threshold)
        result = await strict_host.wait_check_current_step_completions()
        assert result is False

    @pytest.mark.anyio
    async def test_edge_case_no_tasks(self, host_agent):
        """Test edge case where no tasks were started."""
        # Mock the _wait_for_all_tasks_complete method
        host_agent._wait_for_all_tasks_complete = AsyncMock(return_value=True)
        
        # No tasks scenario
        host_agent._total_tasks_in_step = 0
        host_agent._failed_tasks_by_agent = defaultdict(int)
        
        # Should continue (no tasks means no errors)
        result = await host_agent.wait_check_current_step_completions()
        assert result is True

    @pytest.mark.anyio
    async def test_edge_case_all_tasks_failed(self, host_agent):
        """Test edge case where all tasks failed."""
        # Mock the _wait_for_all_tasks_complete method
        host_agent._wait_for_all_tasks_complete = AsyncMock(return_value=True)
        
        # All tasks failed scenario
        host_agent._total_tasks_in_step = 3
        host_agent._failed_tasks_by_agent = defaultdict(int, {"agent_1": 2, "agent_2": 1})
        
        # Should stop (100% > 50% threshold)
        result = await host_agent.wait_check_current_step_completions()
        assert result is False
