# Buttermilk Group Chat Message Types

This document explains the purpose and proper usage of different message types in the Buttermilk group chat flow system. Understanding these distinctions is crucial for maintaining consistency across group chat participants and ensuring proper UI rendering.

## Overview

The Buttermilk framework uses structured message types to ensure consistency in multi-agent group chats. All participants should use appropriate message types based on the nature of their communication rather than sending raw strings.

## Core Message Types

### AgentOutput
**Purpose**: Standard response format for processed outputs from agent execution.

**When to use**:
- Agent has successfully processed input and generated results
- Returning structured data, analysis, or completed tasks
- Normal operation responses

**Key fields**:
- `agent_id`: Identifier of the agent producing the output
- `outputs`: The actual processed results (any type)
- `call_id`: Unique identifier for the execution
- `metadata`: Optional additional information

**Example**:
```python
result = AgentOutput(
    agent_id=self.agent_id,
    outputs="Analysis complete: 5 key findings identified",
    metadata={"analysis_type": "sentiment", "confidence": 0.95}
)
await self._publish(result)
```

### ErrorEvent
**Purpose**: Broadcasting error information within the flow system.

**When to use**:
- System errors (no tools available, configuration issues)
- Processing failures
- Any error condition that needs to be communicated

**Key fields**:
- `source`: Identifier of the component generating the error
- `content`: Error message description
- `call_id`: Unique identifier for the error event
- `is_error`: Always `True` (computed property)

**Example**:
```python
error = ErrorEvent(
    source=self.agent_id,
    content="Unable to process request: no tools available."
)
await self._publish(error)
```

### UIMessage
**Purpose**: Messages sent TO the MANAGER (user/UI) asking for input or feedback.

**When to use**:
- Requesting user confirmation
- Asking user to choose from options
- Seeking clarification or additional input

**Key fields**:
- `content`: Question or information for the user
- `options`: Response options (list of choices, boolean for yes/no, or None for free text)

**Example**:
```python
ui_msg = UIMessage(
    content="Should I proceed with the analysis?",
    options=True  # Yes/No question
)
await self._publish(ui_msg)
```

### ManagerMessage
**Purpose**: Response FROM the MANAGER (user/UI) back to the system.

**When to use**:
- User providing feedback or decisions
- Responding to UIMessage prompts
- User interrupting or controlling flow

**Key fields**:
- `content`: User's response text
- `confirm`: Boolean confirmation/rejection
- `selection`: Chosen option from presented choices
- `halt`: Signal to stop the workflow
- `interrupt`: Signal to pause for review

## Message Type Guidelines

### ❌ Don't Use Raw Strings
```python
# WRONG - Inconsistent with group chat message types
await self._publish("Error: Something went wrong")
```

### ✅ Use Appropriate Structured Types
```python
# CORRECT - Use ErrorEvent for errors
error = ErrorEvent(source=self.agent_id, content="Something went wrong")
await self._publish(error)

# CORRECT - Use AgentOutput for results
result = AgentOutput(agent_id=self.agent_id, outputs="Task completed successfully")
await self._publish(result)
```

### Host Agent Patterns

Host agents coordinate group chats and should follow these patterns:

**For normal coordination**:
```python
response = AgentOutput(
    agent_id=self.agent_id,
    outputs="Coordinating next steps...",
    metadata={"coordination": True}
)
```

**For error conditions**:
```python
error = ErrorEvent(
    source=self.agent_id,
    content="Cannot proceed: insufficient configuration"
)
```

**For user interaction**:
```python
ui_request = UIMessage(
    content="Which agent should handle this task?",
    options=["ANALYZER", "SUMMARIZER", "REVIEWER"]
)
```

## Benefits of Proper Message Types

1. **UI Consistency**: All group chat participants display messages uniformly
2. **Type Safety**: Structured types prevent message format inconsistencies
3. **Better Error Handling**: Errors include proper metadata and are clearly identified
4. **Debugging**: Message types make flow tracking and debugging easier
5. **Future Compatibility**: Structured types support evolving UI requirements

## Common Patterns

### Error Handling in Agents
```python
try:
    # Process task
    result = await self._do_work()
    return AgentOutput(agent_id=self.agent_id, outputs=result)
except Exception as e:
    error = ErrorEvent(source=self.agent_id, content=f"Processing failed: {e}")
    await self._publish(error)
    return None
```

### Host Agent Coordination
```python
# Good: Use AgentOutput for coordination responses
coordination_response = AgentOutput(
    agent_id=self.agent_id,
    outputs=f"Assigned task to {selected_agent}",
    metadata={"assigned_to": selected_agent, "task_type": task_type}
)
await self._publish(coordination_response)
```

### User Confirmation Patterns
```python
# Request confirmation
confirmation_request = UIMessage(
    content="Ready to proceed with batch processing 100 records?",
    options=True
)
await self._publish(confirmation_request)

# Wait for ManagerMessage response with confirm=True/False
```

## Migration Guide

If you find code using raw strings in group chats:

1. **Identify the purpose**: Is it an error, result, or user interaction?
2. **Choose appropriate type**: ErrorEvent for errors, AgentOutput for results, UIMessage for user prompts
3. **Preserve important information**: Move metadata and context into appropriate fields
4. **Test thoroughly**: Ensure UI rendering works correctly with the new message type

## Related Files

- `buttermilk/_core/contract.py`: Message type definitions
- `buttermilk/agents/flowcontrol/`: Host agent implementations
- Tests: Look for `test_*message*` or `test_*agent*` files for examples