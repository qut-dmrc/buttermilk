# AgentOutput Consistency Fix

## Problem
Host agents were sometimes sending raw string messages instead of structured message types (like `AgentOutput` or `UIMessage`). This caused inconsistency in how messages were displayed in the UI, as all participants in a group chat should send messages using the same structured types.

## Root Cause
In `buttermilk/agents/flowcontrol/structured_llmhost.py` line 102, when no tools were available, the agent was publishing a raw string:

```python
await self._publish("Unable to process request: no tools available.")
```

## Solution
Replaced the raw string with a properly structured `AgentOutput` message:

```python
error_response = AgentOutput(
    agent_id=self.agent_id,
    outputs="Unable to process request: no tools available.",
    metadata={"error": True, "reason": "no_tools_available"}
)
await self._publish(error_response)
```

## Benefits
1. **Consistent UI Display**: All host messages now use structured types that display properly in the UI
2. **Better Error Handling**: Error messages include metadata for better debugging
3. **Type Safety**: Eliminates inconsistent message types in the group chat

## Testing
- Created `test_agent_output_consistency.py` to verify the fix
- Comprehensive search confirmed no other instances of raw string publishing in host agents
- Verified all agents in the codebase follow consistent message publishing patterns

## Files Modified
- `buttermilk/agents/flowcontrol/structured_llmhost.py`: Fixed raw string message publishing
- `test_agent_output_consistency.py`: Added test to verify the fix