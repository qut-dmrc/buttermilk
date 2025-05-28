Proceed with Phase 1 (Core Cleanup) of the Robust Session Management System Design Plan.

Robust Session Management System Design Plan

1. Core Architecture Changes

  Session Manager Enhancements

  class SessionManager:
      """
      Centralized session management with:
      - Atomic session operations
      - Graceful shutdown coordination
      - Resource tracking per session
      - WebSocket connection pooling
      """

      # Key additions:
      - session_resources: Dict[str, SessionResources]  # Track all resources
      - active_connections: Dict[str, Set[WebSocket]]  # Multiple connections per session
      - shutdown_handlers: Dict[str, Callable]  # Custom cleanup per session
      - session_locks: Dict[str, asyncio.Lock]  # Prevent race conditions

  Session Lifecycle States

  class SessionStatus(Enum):
      INITIALIZING = "initializing"    # Session created, resources allocating
      ACTIVE = "active"               # Ready for operations
      RECONNECTING = "reconnecting"   # Client disconnected, awaiting reconnect
      TERMINATING = "terminating"     # Cleanup in progress
      TERMINATED = "terminated"       # Cleanup complete
      ERROR = "error"                # Failed state, needs manual cleanup

  2. Session Lifecycle Management

  Creation Flow

  1. Pre-creation validation - Check resource limits, user permissions
  2. Atomic session initialization - Create session with INITIALIZING status
  3. Resource allocation - Track all resources (orchestrators, agents, tasks)
  4. Health check - Verify all components initialized correctly
  5. Activation - Transition to ACTIVE only when fully ready

  Termination Flow

  1. Graceful shutdown signal - Notify all components
  2. Stop accepting new operations - Set TERMINATING status
  3. Cancel active operations - With configurable timeout
  4. Resource cleanup cascade:
    - Cancel agent tasks
    - Stop orchestrator runtime
    - Clear registries and callbacks
    - Close database connections
  5. Final cleanup verification - Ensure no resource leaks
  6. Mark as TERMINATED

  3. WebSocket Reconnection Strategy

  Connection Management

  class WebSocketConnectionPool:
      """
      Manages WebSocket connections with:
      - Automatic reconnection window (configurable, default 30s)
      - Message buffering during disconnects
      - Connection health monitoring
      - Graceful degradation
      """

      async def handle_disconnect(self, session_id: str, ws: WebSocket):
          # 1. Mark connection as disconnected
          # 2. Start reconnection timer
          # 3. Buffer outgoing messages
          # 4. If timer expires, trigger session cleanup

      async def handle_reconnect(self, session_id: str, ws: WebSocket):
          # 1. Verify session still valid
          # 2. Flush buffered messages
          # 3. Resume normal operations

  Client Notification System

  - Heartbeat mechanism every 10 seconds
  - Session status updates on state changes
  - Explicit termination notifications
  - Reconnection instructions with session token

  4. Autogen Orchestrator Lifecycle

  Enhanced Cleanup

  class AutogenOrchestratorEnhanced:
      async def _cleanup(self):
          """Complete cleanup with timeout and verification"""
          try:
              # 1. Stop accepting new messages
              self._accepting_messages = False

              # 2. Cancel all agent tasks
              for agent in self._agent_registry.values():
                  if hasattr(agent, 'cleanup'):
                      await agent.cleanup()

              # 3. Clear registries
              self._agent_registry.clear()
              self._agent_types.clear()

              # 4. Stop runtime with timeout
              cleanup_task = asyncio.create_task(
                  autogen.runtime_stop()
              )
              await asyncio.wait_for(cleanup_task, timeout=10.0)

              # 5. Clear all callbacks
              self._callback_to_groupchat = None

              # 6. Verify cleanup
              await self._verify_cleanup()

          except asyncio.TimeoutError:
              logger.error("Orchestrator cleanup timeout")
              # Force cleanup
              await self._force_cleanup()

  Agent Lifecycle Management

  - Track all async tasks created by agents
  - Implement AgentLifecycleManager to monitor agent health
  - Add timeout for agent operations
  - Graceful degradation if agent becomes unresponsive

  5. State Isolation Improvements

  Session Context Isolation

  class SessionContext:
      """Encapsulates all session-specific state"""
      session_id: str
      user_id: str
      orchestrator: Optional[Orchestrator]
      agents: Dict[str, Agent]
      resources: SessionResources
      message_buffer: MessageBuffer

      def get_isolated_topic(self, base_topic: str) -> str:
          """Generate session-isolated topic names"""
          return f"{self.session_id}:{base_topic}"

  Resource Tracking

  class SessionResources:
      """Track all resources allocated to a session"""
      tasks: Set[asyncio.Task]
      connections: Set[WebSocket]
      file_handles: Set[IO]
      memory_usage: int

      async def cleanup(self) -> CleanupReport:
          """Cleanup all tracked resources with reporting"""

  6. Monitoring and Observability

  Session Health Checks

  class SessionHealthMonitor:
      async def check_session_health(self, session_id: str) -> HealthStatus:
          """
          Periodic health checks including:
          - Memory usage
          - Task status
          - Orchestrator responsiveness
          - WebSocket connection status
          """

  Metrics Collection

  - Session creation/termination rates
  - Average session duration
  - Resource usage per session
  - Cleanup success/failure rates
  - Reconnection statistics

  7. Error Recovery

  Zombie Session Detection

  async def detect_zombie_sessions():
      """
      Identify and cleanup zombie sessions:
      - Sessions with no active connections > threshold
      - Sessions with failed orchestrators
      - Sessions exceeding resource limits
      """

  Recovery Strategies

  1. Soft recovery - Attempt to restore session to healthy state
  2. Partial cleanup - Clean up failed components, keep session active
  3. Full termination - Complete cleanup when recovery fails
  4. Manual intervention - Flag sessions requiring admin attention

  8. Implementation Priority

  1. Phase 1: Core Cleanup (Critical)
    - Fix dual session management issue
    - Implement complete orchestrator cleanup
    - Add proper agent lifecycle management
  2. Phase 2: Robust Lifecycle (High)
    - Implement enhanced session states
    - Add atomic session operations
    - Create resource tracking system
  3. Phase 3: Reconnection (Medium)
    - Implement WebSocket connection pool
    - Add message buffering
    - Create reconnection protocol
  4. Phase 4: Monitoring (Medium)
    - Add health checks
    - Implement metrics collection
    - Create admin dashboard
  5. Phase 5: Advanced Features (Low)
    - Session persistence
    - Cross-server session migration
    - Session replay capabilities