# Flow Control Priority Fixes - Action Plan

## Critical Issues Requiring Immediate Attention

### 1. Error Recovery is Too Aggressive (HIGH PRIORITY)
**Problem**: Most errors lead to session termination, even recoverable ones.

**Fix Required**:
- Implement retry logic for ProcessingError in FlowRunner
- Add exponential backoff for transient failures (network, rate limits)
- Create error recovery strategies per error type

**Implementation Plan**:
```python
# In flowrunner.py run_flow_with_session()
async def run_flow_with_session(self, request, session_id):
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Existing flow execution
            await orchestrator.run(request)
            break
        except ProcessingError as e:
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                logger.warning(f"Retrying after ProcessingError: {e}")
            else:
                raise FatalError(f"Max retries exceeded: {e}")
        except FatalError:
            # No retry for fatal errors
            raise
```

### 2. UI Disconnection Data Loss (HIGH PRIORITY)
**Problem**: When UI disconnects, flow continues but results may be lost if session expires before reconnection.

**Fix Required**:
- Buffer critical messages during disconnection
- Increase grace period for reconnection
- Send buffered messages on reconnection

**Implementation Plan**:
```python
# Add to FlowRunContext
class FlowRunContext:
    # Existing fields...
    message_buffer: list[AgentTrace] = Field(default_factory=list)
    max_buffer_size: int = Field(default=1000)
    
    async def buffer_message_if_disconnected(self, message: AgentTrace):
        if not self.has_active_connections():
            if len(self.message_buffer) < self.max_buffer_size:
                self.message_buffer.append(message)
            else:
                # Drop oldest message
                self.message_buffer.pop(0)
                self.message_buffer.append(message)
```

### 3. Human-in-Loop Toggle Unreliability (HIGH PRIORITY)
**Problem**: Multiple configuration sources without clear precedence make runtime toggling unreliable.

**Fix Required**:
- Establish clear configuration hierarchy
- Add validation when toggling human_in_loop
- Ensure state propagates to all components

**Implementation Plan**:
```python
# Add to HostAgent
class HostAgent:
    def update_human_in_loop(self, new_value: bool, source: str):
        """Update human_in_loop with source tracking"""
        old_value = self.human_in_loop
        self.human_in_loop = new_value
        
        # Log the change
        logger.info(f"human_in_loop changed from {old_value} to {new_value} by {source}")
        
        # Propagate to orchestrator
        if hasattr(self.orchestrator, 'set_human_in_loop'):
            self.orchestrator.set_human_in_loop(new_value)
        
        # Notify UI of mode change
        await self.send_mode_change_notification(new_value)
```

### 4. Session Expiry Too Aggressive (MEDIUM PRIORITY)
**Problem**: 1-hour timeout may be too short for long-running flows with UI interaction.

**Fix Required**:
- Make timeout configurable per flow type
- Extend timeout on activity
- Warn before expiry

**Implementation Plan**:
```python
# In SessionManager
async def extend_session_on_activity(self, session_id: str):
    """Extend session timeout on user activity"""
    session = self.sessions.get(session_id)
    if session and session.status == SessionStatus.ACTIVE:
        session.last_activity = datetime.utcnow()
        
        # Send warning 5 minutes before expiry
        warning_time = session.timeout_seconds - 300
        asyncio.create_task(self._schedule_expiry_warning(session_id, warning_time))
```

## Implementation Priority Order

### Phase 1: Prevent Data Loss (1-2 days)
1. Implement message buffering during disconnection
2. Add flush buffer on reconnection
3. Extend default session timeout to 2 hours
4. Add activity-based timeout extension

### Phase 2: Improve Error Handling (2-3 days)
1. Add retry logic for ProcessingError
2. Implement exponential backoff
3. Create error categorization (retryable vs non-retryable)
4. Add circuit breaker for external services

### Phase 3: Fix Configuration Issues (1-2 days)
1. Document configuration hierarchy
2. Add validation for human_in_loop changes
3. Ensure propagation to all components
4. Add configuration change logging

### Phase 4: Testing & Validation (2-3 days)
1. Add integration tests for disconnection scenarios
2. Test error recovery paths
3. Validate configuration changes
4. Load test with multiple concurrent sessions

## Quick Wins (Can implement immediately)

### 1. Increase Session Timeout
```python
# In flowrunner.py
DEFAULT_SESSION_TIMEOUT = 7200  # 2 hours instead of 1
```

### 2. Add Connection State Logging
```python
# In monitor_ui()
logger.info(f"WebSocket disconnected for session {session_id}, entering RECONNECTING state")
logger.info(f"Session {session_id} has {len(active_connections)} remaining connections")
```

### 3. Validate State Transitions
```python
# In SessionManager
def validate_transition(self, session: FlowRunContext, new_status: SessionStatus):
    valid = session.can_transition_to(new_status)
    if not valid:
        logger.error(f"Invalid transition from {session.status} to {new_status}")
    return valid
```

## Testing Strategy

### Unit Tests Required
- Test message buffering during disconnection
- Test retry logic with different error types
- Test configuration precedence
- Test session timeout extension

### Integration Tests Required
- Simulate UI disconnection during flow execution
- Test reconnection with buffered messages
- Test error recovery with retries
- Test human_in_loop toggle during execution

### Manual Testing Scenarios
1. Start long-running flow, disconnect UI, reconnect after 30 mins
2. Trigger ProcessingError and verify retry behavior
3. Toggle human_in_loop during flow execution
4. Test multiple UI connections to same session

## Success Metrics
- Zero data loss during UI disconnections
- 90% of ProcessingErrors recovered without session termination
- 100% reliable human_in_loop toggling
- Session expiry warnings delivered 5 minutes before timeout

## Notes for Developers
- All changes should maintain backward compatibility
- Add comprehensive logging for debugging
- Update documentation for new behaviors
- Consider feature flags for gradual rollout