# Investigation Notes: Robust Flow Execution with Resumable UI Sessions
Core Architecture Questions
## Flow State Management

- How is flow state persisted between UI disconnections?
- Can flows checkpoint their progress for resumption?
- What happens to intermediate results when UI disconnects?

## Error Classification System

Need to distinguish between:
- Recoverable errors: Network timeouts, temporary service unavailability
- Fatal errors: Invalid input, authentication failures, resource exhaustion
- UI-dependent errors: Human input required, approval needed
- Current code treats all exceptions as fatal

## Session Lifecycle Management

- When should sessions be created/destroyed?
- How to handle orphaned sessions (UI never reconnects)?
- Session expiry vs. flow completion timing
- Specific Technical Investigations

## FlowRunner Task Management
- Can `run_flow()` tasks be paused/resumed?
- How to detect if a flow requires UI interaction vs. autonomous execution?
- Should tasks have different cancellation policies based on flow type?

## Session State Persistence
- Where is session state stored? (memory, database, cache)
- Can UI reconnect to in-progress flows?
- How to handle concurrent UI connections to same session?

## Error Recovery Mechanisms
- Retry logic for transient failures
- Circuit breaker patterns for external services
- Graceful degradation when UI unavailable

# Proposed Investigation Areas
## Flow Execution Modes

Batch Mode: Flow runs to completion without UI (implemented)
Interactive Mode: Flow requires periodic UI input (implemented)
Hybrid Mode: Flow can switch between autonomous/interactive? (partially implemneted; client can toggle human_in_loop but it's not reliable)

## Error Handling Framework

- Separate task cancellation policies by error type. FatalError is fatal; processingerror might be recoverable. Check that this is both correctly implemented and respected. 
- Background task monitoring and cleanup
- Resource leak prevention

# Recommended Next Steps
Audit FlowRunner class for existing error handling patterns
Map flow types to understand UI dependency requirements
Audit session management and flow control

## Additional instructions

- Only address back-end design. 
- DO NOT investigate the front-end implementations (including console.py and /frontend/chat).
- DOCUMENT WHAT YOU LEARN in a concise way that will help LLM developers more efficiently understand the code base.


##  Questions for Architecture Review
Should flows be designed as resumable by default?
What's the acceptable window for UI reconnection?
How to handle partial flow results during errors?

## Advanced Features (Low priority)
    - Add health checks
    - Implement metrics collection
    - Create admin dashboard
    - Session persistence
    - Cross-server session migration
    - Session replay capabilities