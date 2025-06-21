# Buttermilk Integration Testing Framework

## Overview

This framework enables testing of complex, long-running flows with real user interaction, minimal mocking, and support for both backend (WebSocket/MCP) and frontend (SvelteKit) testing.

## Design Principles

1. **Respect Natural Flow Timing**: Wait for orchestrator signals, not arbitrary delays
2. **Use Real Components**: Test actual flows with live code
3. **Event-Driven Testing**: React to flow events rather than forcing timing
4. **Incremental Complexity**: Start simple, add features as needed
5. **Clear Failure Messages**: Know exactly what went wrong and where

## Architecture

### Core Components

1. **FlowTestClient**: WebSocket client that understands flow lifecycle
2. **MessageCollector**: Captures and categorizes all flow messages
3. **FlowEventWaiter**: Waits for specific events with timeouts
4. **TestFlowRunner**: Manages test server lifecycle
5. **ResponseSimulator**: Provides contextual responses to prompts

### Message Flow

```
Test Case -> FlowTestClient -> WebSocket -> FlowRunner -> Orchestrator
    ^                                                           |
    |                                                           v
    +----------- MessageCollector <-- UI Callback <-- Agents/Host
```

## Implementation Phases

### Phase 1: Core WebSocket Testing (Week 1)
- FlowTestClient with event-based waiting
- MessageCollector for comprehensive logging
- Basic test for OSB flow
- Test configuration management

### Phase 2: Advanced Flow Testing (Week 2)
- Multi-turn conversation testing
- Error handling and recovery
- Parallel test execution
- Performance benchmarking

### Phase 3: Frontend Integration (Week 3)
- Playwright setup for SvelteKit
- Coordinated backend/frontend tests
- Visual regression testing
- Accessibility testing

### Phase 4: MCP & Production (Week 4)
- MCP protocol testing
- CI/CD integration
- Test reporting dashboard
- Documentation and training

## Quick Start

```python
async def test_osb_flow():
    async with FlowTestClient.create() as client:
        # Start flow and wait for initialization
        await client.start_flow("osb", "What is Meta's hate speech policy?")
        
        # Wait for host's greeting and initial prompt
        prompt = await client.wait_for_ui_message(
            pattern="proceed|confirm|start",
            timeout=30
        )
        
        # Send natural response
        await client.send_manager_response("Yes, please proceed")
        
        # Wait for agents to work
        results = await client.wait_for_agent_results(
            expected_agents=["researcher", "policy_analyst"],
            timeout=120
        )
        
        # Verify results
        assert any("hate speech" in r.content for r in results)
```

## Test Configuration

Tests use a special configuration that:
- Disables Weave tracing for speed
- Uses deterministic LLMs for reproducibility
- Isolates storage to prevent conflicts
- Enables comprehensive logging

See `conf/test/` for test-specific configurations.