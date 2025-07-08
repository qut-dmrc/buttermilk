#!/usr/bin/env python
"""Minimal test to check AutogenOrchestrator message flow."""

import asyncio
import json
import pytest
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk._core.types import RunRequest
from buttermilk._core.contract import FlowEvent
from buttermilk._core.bm_init import bm_init, set_bm

# Store all messages sent to UI
ui_messages = []

async def mock_send_to_ui(message):
    """Mock callback to capture messages."""
    print(f"[MOCK UI] Received message: {type(message).__name__}")
    if hasattr(message, 'content'):
        print(f"[MOCK UI]   Content: {message.content[:100]}")
    ui_messages.append(message)

async def test_orchestrator_setup():
    """Test that AutogenOrchestrator sends setup message."""
    # Initialize buttermilk
    bm = bm_init()
    set_bm(bm)
    
    # Create a minimal flow config
    from omegaconf import OmegaConf
    flow_config = OmegaConf.create({
        "orchestrator": "buttermilk.orchestrators.groupchat.AutogenOrchestrator",
        "agents": {
            "researcher": {
                "role": "researcher",
                "agent": "buttermilk.agents.rag.RAGAgent",
                "description": "Research agent",
                "variants": [{"model": "gemini-2.0-flash-exp"}]
            }
        },
        "observers": {},
        "parameters": {}
    })
    
    # Create flow runner
    flow_runner = FlowRunner(flows={"test": flow_config})
    
    # Create run request
    run_request = RunRequest(
        flow="test",
        prompt="Test prompt",
        ui_type="test",
        callback_to_ui=mock_send_to_ui
    )
    
    # Run the flow
    print("[TEST] Starting flow...")
    task = asyncio.create_task(flow_runner.run_flow(run_request, wait_for_completion=False))
    
    # Wait a bit for messages
    await asyncio.sleep(3)
    
    # Check messages
    print(f"\n[TEST] Received {len(ui_messages)} messages:")
    for i, msg in enumerate(ui_messages):
        print(f"  {i+1}. {type(msg).__name__}")
        if isinstance(msg, FlowEvent) and "AutogenOrchestrator" in str(msg.content):
            print(f"     FOUND: {msg.content}")
            
    # Cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Check if we got the expected message
    found_setup_msg = False
    for msg in ui_messages:
        if isinstance(msg, FlowEvent) and "Setting up AutogenOrchestrator" in str(msg.content):
            found_setup_msg = True
            break
    
    assert found_setup_msg, "Did not receive 'Setting up AutogenOrchestrator' message"
    print("\n[TEST] SUCCESS: Found setup message!")

if __name__ == "__main__":
    asyncio.run(test_orchestrator_setup())