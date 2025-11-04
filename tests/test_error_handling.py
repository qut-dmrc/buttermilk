"""Test script to validate error handling in HostAgent.

This script creates a minimal test to verify that the HostAgent
properly tracks errors and stops flow execution when too many tasks fail.
"""

import asyncio
import sys
from unittest.mock import MagicMock

import pytest
from autogen_core import MessageContext

from buttermilk._core.contract import TaskProcessingComplete, TaskProcessingStarted
from buttermilk.agents.flowcontrol.host import HostAgent


@pytest.mark.anyio
async def test_error_handling():
    """Test that HostAgent stops flow when error threshold is exceeded."""
    print("Testing HostAgent error handling...")

    # Create a host agent with a low error threshold for testing
    host = HostAgent(
        role="HOST",
        description="Test host agent",
        parameters={"human_in_loop": False, "error_threshold": 0.4},  # 40% error threshold
        unique_identifier="test_host",
    )

    # Mock message context
    mock_ctx = MagicMock(spec=MessageContext)
    mock_ctx.sender = None

    # Simulate 5 tasks starting
    print("Simulating 5 tasks starting...")
    for i in range(5):
        start_msg = TaskProcessingStarted(agent_id=f"agent_{i}", role="WORKER", task_index=0)
        await host.handle_task_started(start_msg, mock_ctx)

    # Simulate 3 tasks completing with errors (60% failure rate)
    print("Simulating 3 tasks completing with errors...")
    for i in range(3):
        complete_msg = TaskProcessingComplete(
            agent_id=f"agent_{i}",
            role="WORKER",
            task_index=0,
            is_error=True,  # This task failed
        )
        await host.handle_task_complete(complete_msg, mock_ctx)

    # Simulate 2 tasks completing successfully
    print("Simulating 2 tasks completing successfully...")
    for i in range(3, 5):
        complete_msg = TaskProcessingComplete(
            agent_id=f"agent_{i}",
            role="WORKER",
            task_index=0,
            is_error=False,  # This task succeeded
        )
        await host.handle_task_complete(complete_msg, mock_ctx)

    # Check error tracking
    print(f"Failed tasks by agent: {dict(host._failed_tasks_by_agent)}")
    print(f"Total tasks in step: {host._total_tasks_in_step}")

    # Test wait_check_current_step_completions - should return False due to high error rate
    print("Testing wait_check_current_step_completions...")
    result = await host.wait_check_current_step_completions()

    if not result:
        print("✅ SUCCESS: Host correctly stopped flow due to high error rate (60% > 40% threshold)")
    else:
        print("❌ FAILURE: Host should have stopped flow but didn't")

    return result


@pytest.mark.anyio
async def test_error_handling_below_threshold():
    """Test that HostAgent continues when error rate is below threshold."""
    print("\nTesting HostAgent with error rate below threshold...")

    # Create a host agent with a higher error threshold
    host = HostAgent(
        role="HOST",
        description="Test host agent",
        parameters={"human_in_loop": False, "error_threshold": 0.7},  # 70% error threshold
        unique_identifier="test_host2",
    )

    # Mock message context
    mock_ctx = MagicMock(spec=MessageContext)
    mock_ctx.sender = None

    # Simulate 5 tasks starting
    print("Simulating 5 tasks starting...")
    for i in range(5):
        start_msg = TaskProcessingStarted(agent_id=f"agent_{i}", role="WORKER", task_index=0)
        await host.handle_task_started(start_msg, mock_ctx)

    # Simulate 2 tasks completing with errors (40% failure rate)
    print("Simulating 2 tasks completing with errors...")
    for i in range(2):
        complete_msg = TaskProcessingComplete(
            agent_id=f"agent_{i}",
            role="WORKER",
            task_index=0,
            is_error=True,  # This task failed
        )
        await host.handle_task_complete(complete_msg, mock_ctx)

    # Simulate 3 tasks completing successfully
    print("Simulating 3 tasks completing successfully...")
    for i in range(2, 5):
        complete_msg = TaskProcessingComplete(
            agent_id=f"agent_{i}",
            role="WORKER",
            task_index=0,
            is_error=False,  # This task succeeded
        )
        await host.handle_task_complete(complete_msg, mock_ctx)

    # Check error tracking
    print(f"Failed tasks by agent: {dict(host._failed_tasks_by_agent)}")
    print(f"Total tasks in step: {host._total_tasks_in_step}")

    # Test wait_check_current_step_completions - should return True since error rate is below threshold
    print("Testing wait_check_current_step_completions...")
    result = await host.wait_check_current_step_completions()

    if result:
        print("✅ SUCCESS: Host correctly continued flow (40% ≤ 70% threshold)")
    else:
        print("❌ FAILURE: Host should have continued flow but stopped")

    return result


if __name__ == "__main__":

    async def main():
        try:
            # Test high error rate (should stop)
            high_error_result = await test_error_handling()

            # Test low error rate (should continue)
            low_error_result = await test_error_handling_below_threshold()

            if not high_error_result and low_error_result:
                print("\n🎉 ALL TESTS PASSED: Error handling works correctly!")
                return True
            else:
                print("\n❌ SOME TESTS FAILED")
                return False
        except Exception as e:
            print(f"Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            return False

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
