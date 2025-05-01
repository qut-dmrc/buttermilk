# Agent Contract Changes

## Overview

The agent contract in Buttermilk has been updated to improve reliability and traceability. These changes primarily affect:
1. How agents return responses from their `_process` method
2. How agent traces are created and managed
3. Parameter handling for agent execution

## Key Changes

### 1. AgentResponse as the Standard Return Type

All agents' `_process` methods must now return an `AgentResponse` object, which contains both outputs and metadata. This standardizes the agent interface and makes it easier to build composable agents.

```python
# Before
async def _process(self, *, message: AgentInput, cancellation_token=None, **kwargs) -> AgentTrace:
    # Processing logic
    return AgentTrace(agent_info=self._cfg, outputs=result)

# After
async def _process(self, *, message: AgentInput, cancellation_token=None, **kwargs) -> AgentResponse:
    # Processing logic
    return AgentResponse(metadata={"source": self.id}, outputs=result)
```

### 2. Mandatory AgentInfo in AgentTrace

The `agent_info` field in `AgentTrace` is now mandatory, ensuring all traces contain proper agent identification and configuration information. This is handled automatically by the base `Agent.__call__` method, which creates the trace with agent_info from `self._cfg`.

```python
# Before
trace = AgentTrace()  # agent_info was optional

# After
trace = AgentTrace(
    agent_info=self._cfg,  # Now required
    inputs=final_input
)
```

### 3. Simplified Return Flow

The base `Agent.__call__` method now:
1. Creates the `AgentTrace` with all required fields
2. Calls the agent's `_process` method to get an `AgentResponse`
3. Transfers outputs and metadata from the `AgentResponse` to the `AgentTrace`
4. Returns the trace to the orchestrator

This allows subclasses to focus on producing outputs and metadata without worrying about trace management.

## Migration Guide

### For Agent Implementations

1. Update your `_process` method signature to return `AgentResponse`:
   ```python
   from buttermilk._core.agent import AgentResponse
   
   async def _process(self, *, message: AgentInput, cancellation_token=None, **kwargs) -> AgentResponse:
       # Your processing logic here
       return AgentResponse(
           metadata={"source": self.id, "role": self.role}, 
           outputs=your_result
       )
   ```

2. If your agent previously returned different message types directly from `_process`, wrap them in an `AgentResponse`:
   ```python
   # Before
   return StepRequest(role=WAIT, content="Waiting for input")
   
   # After
   return AgentResponse(
       metadata={"source": self.id}, 
       outputs=StepRequest(role=WAIT, content="Waiting for input")
   )
   ```

3. If you have custom `evaluate_content` methods that return `AgentTrace`, update them to create a trace from the response:
   ```python
   # Before
   result: AgentTrace = await self._process(message=message)
   return result
   
   # After
   response = await self._process(message=message)
   trace = AgentTrace(
       agent_info=self._cfg,
       inputs=message,
       outputs=response.outputs,
       metadata=response.metadata
   )
   return trace
   ```

### For Flow Orchestrators

Orchestrators should be unaffected as they receive `AgentTrace` objects as before. The internal `AgentResponse` is only used between `__call__` and `_process`.

## Benefits

- Better traceability: Every trace now has complete agent information
- Cleaner separation of concerns: Agents focus on producing outputs, the base class handles trace management
- More consistent error handling: The standard pattern makes error detection and recovery more predictable
- Future extensibility: The AgentResponse pattern allows for adding more metadata and output types without breaking changes
