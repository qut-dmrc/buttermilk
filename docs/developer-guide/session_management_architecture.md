# Buttermilk Session Management & Flow Execution Architecture

## Overview
Buttermilk implements an in-memory session management system for flow execution with support for UI disconnection/reconnection within the same server lifetime. The architecture is sophisticated but lacks persistent storage and flow checkpointing capabilities.

## Key Components

### FlowRunner (`buttermilk/runner/flowrunner.py`)
- Main entry point for flow execution
- Manages `SessionManager` singleton
- Handles flow lifecycle from initialization to cleanup

### SessionManager
- In-memory session storage: `dict[str, FlowRunContext]`
- Atomic operations with session locks
- Automatic cleanup of expired sessions (1 hour default timeout)
- Background cleanup task runs every 5 minutes

### FlowRunContext
- Encapsulates all state for a single flow run
- Tracks messages, progress, websockets, and resources
- Session states: INITIALIZING → ACTIVE → RECONNECTING/COMPLETED/ERROR → TERMINATING → TERMINATED

## Flow Execution Modes

### 1. Batch Mode (`conf/run/batch.yaml`)
- `human_in_loop: false`
- Autonomous execution without UI interaction
- Flows run to completion independently

### 2. Interactive Mode (`conf/run/console.yaml`, `conf/run/api.yaml`)
- `human_in_loop: true`
- Requires periodic UI input/confirmation
- HostAgent waits for user responses

### 3. Hybrid Mode (Partially Implemented)
- Runtime toggle via `ManagerMessage`
- Reliability issues noted in session plan
- Multiple configuration sources without clear precedence

## Error Classification & Handling

### Error Types (`buttermilk/_core/exceptions.py`)
- **FatalError**: Unrecoverable, terminates session immediately
- **ProcessingError**: Non-fatal, orchestrator decides recovery
- **RateLimit**: API limits, may retry with backoff
- **ProcessingFinished/NoMoreResults**: Control flow signals

### Error Recovery Patterns
```python
# ProcessingError - Continue with recovery
except ProcessingError as e:
    logger.error(f"Non-fatal error: {e}")
    # Orchestrator decides recovery strategy

# FatalError - Immediate termination
except FatalError:
    session.status = SessionStatus.ERROR
    raise
```

## Session Lifecycle Management

### Resource Tracking
```python
class SessionResources:
    tasks: set[asyncio.Task]
    websockets: set[Any]
    file_handles: set[Any]
    memory_usage: int
    custom_resources: dict[str, Any]
```

### Cleanup Process
1. Cancel all tracked tasks (with timeout)
2. Close WebSockets gracefully
3. Close file handles
4. Cleanup custom resources
5. Return cleanup report

### Orphaned Session Handling
- Sessions expire after 1 hour of inactivity
- Completed sessions cleaned after 5 minute grace period
- Sessions with no connections cleaned after 30 minutes
- ERROR state sessions cleaned immediately

## UI Disconnection/Reconnection

### Disconnection Handling
- WebSocket disconnect caught in `monitor_ui()`
- Session transitions to RECONNECTING state
- Flow continues in background
- Resources maintained

### Reconnection Support
- Client reconnects using same session_id
- `get_or_create_session()` retrieves existing session
- Multiple concurrent connections supported per session
- **Limitation**: Only works while server is running (no persistence)

## Critical Architecture Gaps

1. **No Persistent Storage**
   - All session state in-memory
   - Lost on server restart
   - No database/cache layer

2. **No Flow Checkpointing**
   - Flows cannot resume from intermediate states
   - No state serialization/deserialization

3. **Limited Error Recovery**
   - Most errors lead to termination
   - No retry mechanisms for flows
   - No circuit breaker patterns

4. **Configuration Inconsistencies**
   - `human_in_loop` exists at multiple levels
   - Runtime toggling unreliable
   - No clear configuration precedence

## Recommendations for Developers

### When Adding New Features
1. Follow existing session state transition rules
2. Register all resources with SessionResources for cleanup
3. Use ProcessingError for recoverable errors, FatalError for critical
4. Respect the `human_in_loop` configuration hierarchy

### Common Patterns
```python
# Resource registration
session.resources.tasks.add(task)
session.resources.custom_resources["my_resource"] = resource

# Error handling
if recoverable_condition:
    raise ProcessingError("Can recover from this")
else:
    raise FatalError("Cannot continue")

# Session state transitions
if session.can_transition_to(new_status):
    session.status = new_status
```

### Testing Considerations
- Test UI disconnection/reconnection scenarios
- Verify resource cleanup on all error paths
- Check session expiry handling
- Test concurrent connections to same session

## Future Enhancement Opportunities
1. Add persistent session storage (Redis/PostgreSQL)
2. Implement flow checkpointing for resumption
3. Add retry mechanisms with exponential backoff
4. Implement circuit breakers for external services
5. Create clear configuration hierarchy for `human_in_loop`
6. Add session replay capabilities
7. Implement cross-server session migration