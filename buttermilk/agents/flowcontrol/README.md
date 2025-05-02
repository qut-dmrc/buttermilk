# Flow Control Agents

This directory contains agents responsible for controlling the flow of conversations in agent group chats.

## Overview

The flow control architecture in Buttermilk has been refactored to delegate conversation flow decisions to specialized host agents. This design:

1. Decouples flow control logic from the technical orchestration implementation
2. Enables more sophisticated conversation patterns and exploration strategies
3. Allows different flow control strategies to be swapped in/out without changing orchestrators
4. Provides more direct control over agent execution and interaction

## Components

### Base Host Agent

The `LLMHostAgent` serves as the foundation for flow control, providing:

- Sequence generation and control
- Participant tracking and step completion detection
- Direct step execution via topic-based messaging
- Human-in-the-loop confirmation capability
- Context building for decision making

### Explorer Host

The `ExplorerHost` specializes in guided exploration of conversational paths:

- Maintains history of explored paths and their outcomes
- Prioritizes unexplored conversation branches
- Incorporates user feedback into decision making
- Optimizes for discovery and coverage of agent capabilities

### Sequential Batch Host

The `SequentialBatchHost` enables automated batch processing:

- Executes a predefined sequence of steps without user intervention
- Handles error recovery and retries
- Provides deterministic execution for automation and testing
- Minimizes overhead for production batch jobs

## Usage

Orchestrators (like `AutogenOrchestrator` or `Selector`) now act primarily as message buses, while hosts control the conversational flow:

```python
# Configure a flow with an explorer host
flow_config = {
    "agents": {
        "HOST": {
            "agent": "ExplorerHost",
            "parameters": {
                "exploration_mode": "interactive", 
                "prioritize_unexplored": True
            }
        },
        "ANALYST": {...},
        "CRITIC": {...}
    }
}

# Run with any orchestrator - the host controls the flow
orchestrator = AutogenOrchestrator(**flow_config)
await orchestrator.run()
```

## Design Philosophy

1. **Separation of Concerns**: 
   - Orchestrators handle technical execution (agent registration, message routing)
   - Host agents handle flow decisions (what happens next, sequence management)
   - Agent implementations handle domain logic (generating content, analyzing data)

2. **Delegation Over Control**:
   - The orchestrator delegates flow decisions to the host
   - The host can execute steps directly or delegate back to the orchestrator
   - This creates flexibility in implementation and extension

3. **Direct Communication**:
   - Host agents can directly message other agents using topic-based publishing
   - This reduces complexity in the orchestrator's execution loop
   - It allows for more sophisticated interaction patterns

4. **Human Integration**:
   - Hosts can decide when human confirmation is needed
   - Different hosts can implement different levels of human involvement
   - The system can be adjusted from fully automated to highly interactive

## Implementation Details

The host agents now have direct execution capability via:

```python
async def _execute_step(self, step: StepRequest) -> None:
    # Create the message for the target agent
    message = AgentInput(prompt=step.prompt, records=self._records)
    
    # Publish directly to agent role topic
    if self._input_callback:
        await self._input_callback(message)
```

This allows bypassing the orchestrator's execution loop for more efficient agent-to-agent communication.

## Future Work

- Create additional host types for different conversation patterns
- Enhance execution to support fan-out/fan-in parallel agent execution
- Improve context tracking for multi-turn conversations between specific agents
- Add visualization tools for exploration paths and decision making
