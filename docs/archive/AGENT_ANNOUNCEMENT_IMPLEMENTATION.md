# Agent Announcement System Implementation

This document describes the Agent Announcement system implemented for Issue #87.

## Overview

The Agent Announcement system allows agents to dynamically announce their presence, capabilities, and status when joining or leaving a groupchat flow. This enables:

- Dynamic agent discovery
- Real-time capability tracking
- Visual feedback in UI
- Better coordination in multi-agent flows

## Implementation Details

### 1. AgentAnnouncement Message Type

Added to `buttermilk/_core/contract.py`:

```python
class AgentAnnouncement(FlowEvent):
    """Agent self-announcement message containing configuration and capabilities."""
    agent_config: AgentConfig
    available_tools: list[str]
    supported_message_types: list[str]
    status: Literal["joining", "active", "leaving"]
    announcement_type: Literal["initial", "response", "update"]
    responding_to: str | None = None
```

### 2. Base Agent Methods

Added to `buttermilk/_core/agent.py`:

- `send_announcement()`: Sends announcement to public callback
- `_create_announcement()`: Creates announcement message
- Auto-response to host announcements in `_listen()` method

### 3. Host Agent Registry

Enhanced `buttermilk/agents/flowcontrol/host.py`:

- Thread-safe registries using `PrivateAttr` and `asyncio.Lock`
- `_agent_registry`: Tracks agent announcements
- `_tool_registry`: Maps tools to agents
- Registry summary generation for UI display

### 4. Orchestrator Integration

Updated `buttermilk/orchestrators/groupchat.py`:

- Broadcasts initialization event on startup
- Handles AgentAnnouncement messages
- Routes announcements to HOST topic

### 5. UI Enhancements

Updated `buttermilk/agents/ui/console.py`:

- Displays agent announcements with status colors
- Shows available tools
- `!agents` command to list active agents
- IRC-style formatting for agent registry

## Usage Example

When an OSB flow starts:

```
[01:22:16] 🔬 🧑‍💻 ENHANCED_RES │ 🆕 [JOINING] ENHANCED_RESEARCHER Agent
[01:22:16] 💬 POLICY_ANALYST   │ 🆕 [JOINING] POLICY_ANALYST Agent
[01:22:16] 💬 FACT_CHECKER     │ 🆕 [JOINING] FACT_CHECKER Agent
[01:22:16] 💬 RESEARCH_EXPLORE │ 🆕 [JOINING] RESEARCH_EXPLORER Agent
[01:22:16] 💬 HOST             │ 🆕 [JOINING] HOST Agent
```

Users can type `!agents` to see the current registry:

```
=== Active Agents (4) ===
🟢 ENHANCED_RESEARCHER [ACTIVE] - Advanced research assistant
   Tools: search
🟢 POLICY_ANALYST [ACTIVE] - Policy analysis assistant
   Tools: search
...
```

## Testing

Comprehensive test coverage added:

- 42 tests for agent announcement functionality
- Thread safety validation
- Message routing verification
- UI display testing

## Benefits

1. **Dynamic Discovery**: Agents can join/leave flows dynamically
2. **Capability Awareness**: Other agents know what tools are available
3. **User Visibility**: Clear feedback about active agents
4. **Better Coordination**: Host can make informed decisions

## Future Enhancements

- WebSocket events for real-time UI updates
- Agent capability negotiation
- Performance metrics per agent
- Agent health monitoring