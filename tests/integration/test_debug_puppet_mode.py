"""Test and demonstrate DebugAgent puppet mode functionality."""

import asyncio

import pytest

from buttermilk.debug.debug_agent import DebugAgent


@pytest.mark.slow
@pytest.mark.anyio
async def test_debug_agent_puppet_mode():
    """Demonstrate the DebugAgent puppet mode for flow debugging."""

    # Initialize the DebugAgent
    debug_agent = DebugAgent(agent_name="debug_agent", role="Debugging Assistant")

    print("\n=== Starting DebugAgent Puppet Mode Demo ===\n")

    try:
        # Step 1: Start puppet mode
        print("1. Starting puppet mode connection to localhost:8000...")
        await debug_agent.start_puppet_mode(host="localhost", port=8000)
        print("✓ Puppet mode started successfully\n")

        # Step 2: Start a flow
        print("2. Starting 'trans' flow with initial prompt...")
        await debug_agent.puppet_start_flow(
            flow_name="trans",
            prompt="What are the key themes in critical thinking education?",
            record="test_record_001",
            criteria="critical thinking",
        )
        print("✓ Flow started successfully\n")

        # Step 3: Monitor messages for a few seconds
        print("3. Monitoring flow messages...")
        await asyncio.sleep(3)

        # Get recent messages
        messages = debug_agent.puppet_get_messages(last_n=10, message_type="ui_message")
        print(f"   Received {len(messages)} UI messages")

        # Display key messages
        for i, msg in enumerate(messages[-3:], 1):  # Show last 3 messages
            print(
                f"   Message {i}: {msg.get('type', 'unknown')} - {str(msg.get('content', ''))[:100]}..."
            )

        # Step 4: Get flow summary
        print("\n4. Getting flow summary...")
        summary = debug_agent.puppet_get_summary()
        print(f"   Flow State: {summary.get('flow_state', 'unknown')}")
        print(f"   Total Messages: {summary.get('total_messages', 0)}")
        print(f"   Active Agents: {summary.get('active_agents', [])}")

        # Step 5: Send a response if needed
        if summary.get("flow_state") == "waiting_for_input":
            print("\n5. Sending response to flow...")
            await debug_agent.puppet_send_response(
                "Please focus on pedagogical approaches"
            )
            print("✓ Response sent successfully")

            # Wait for processing
            await asyncio.sleep(2)

            # Check updated state
            updated_summary = debug_agent.puppet_get_summary()
            print(
                f"   Updated Flow State: {updated_summary.get('flow_state', 'unknown')}"
            )

        # Step 6: Monitor for completion
        print("\n6. Monitoring for flow completion...")
        max_wait = 30  # Maximum 30 seconds
        for i in range(max_wait):
            await asyncio.sleep(1)
            current_summary = debug_agent.puppet_get_summary()
            if current_summary.get("flow_state") == "completed":
                print(f"✓ Flow completed after {i + 1} seconds")
                break
            if i % 5 == 0:
                print(f"   Still processing... ({i} seconds)")

        # Final summary
        final_summary = debug_agent.puppet_get_summary()
        print("\n=== Final Flow Summary ===")
        print(f"Flow State: {final_summary.get('flow_state', 'unknown')}")
        print(f"Total Messages: {final_summary.get('total_messages', 0)}")
        print(f"Agents Involved: {final_summary.get('active_agents', [])}")

        # Get completion messages
        final_messages = debug_agent.puppet_get_messages(
            last_n=5, message_type="ui_message"
        )
        if final_messages:
            print("\nFinal agent outputs:")
            for msg in final_messages[-2:]:
                if "agent" in str(msg).lower():
                    print(
                        f"  - {msg.get('type', 'unknown')}: {str(msg.get('content', ''))[:150]}..."
                    )

        print("\n✓ Debug Agent Puppet Mode demonstration completed successfully!")

    except Exception as e:
        print(f"\n✗ Error during puppet mode demo: {e}")
        raise

    finally:
        # Clean up
        print("\n7. Stopping puppet mode...")
        await debug_agent.stop_puppet_mode()
        print("✓ Puppet mode stopped\n")


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_debug_agent_puppet_mode())
