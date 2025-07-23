#!/usr/bin/env python3
"""Script to create GitHub issue for ws_debug_cli migration challenges"""

import subprocess
import sys

def create_issue():
    title = "Migrate ws_debug_cli and debug_agent to standalone tools"
    
    body = """## Overview

During the MCP debugging tools reorganization (#116), we identified that two existing debug tools have tight coupling with Buttermilk's internal code that prevents easy migration to standalone scripts.

## Affected Components

### 1. `buttermilk/debug/ws_debug_cli.py`
- **Dependencies**: 
  - `from buttermilk.agents.test_utils import FlowTestClient`
  - Relies on internal WebSocket protocol implementation
  - Uses Buttermilk's message models and session management

### 2. `buttermilk/debug/debug_agent.py`
- **Dependencies**:
  - `from buttermilk._core import AgentInput, logger`
  - `from buttermilk._core.agent import Agent`
  - `from buttermilk._core.contract import AgentOutput`
  - `from buttermilk.agents.test_utils import FlowTestClient`
  - Inherits from Agent base class
  - Uses internal runtime and registration system

## Challenges

1. **FlowTestClient Dependency**: Both tools rely on `FlowTestClient` which is deeply integrated with Buttermilk's:
   - Message models (AgentMessage, UserMessage, etc.)
   - WebSocket protocol implementation
   - Session management logic

2. **Agent Framework**: The debug_agent is implemented as a proper Buttermilk agent, requiring:
   - Agent registration system
   - Runtime context
   - Message passing infrastructure

3. **Protocol Knowledge**: The tools need intimate knowledge of:
   - WebSocket message formats
   - Flow execution protocol
   - Session lifecycle management

## Potential Solutions

### Option 1: Extract Minimal Protocol Library
Create a minimal `buttermilk-protocol` package containing:
- Message type definitions
- WebSocket protocol specification
- Basic client implementation

### Option 2: Keep as Internal Tools
Accept that these are internal debugging tools that require Buttermilk installation:
- Document the dependency requirement
- Keep them in `buttermilk/debug/`
- Provide standalone alternatives for common use cases

### Option 3: REST API Bridge
Create REST API endpoints that expose the functionality:
- `/api/debug/flows/start`
- `/api/debug/flows/send`
- `/api/debug/logs/read`
- Then create standalone scripts that use these endpoints

## Recommendation

For now, we've created `scripts/mcp_debug/websocket_debug.py` as a standalone alternative that reimplements the basic WebSocket client functionality without Buttermilk dependencies. 

The original tools should remain as "advanced debugging tools" that require Buttermilk installation, while the standalone scripts provide basic functionality for MCP integration.

## Action Items

- [ ] Document the dependency requirements for advanced debug tools
- [ ] Consider creating a protocol specification document
- [ ] Evaluate if REST API bridge approach is worth implementing
- [ ] Update MCP tool configurations to use standalone scripts where possible

---
Created as part of #116 implementation"""

    # Create the issue using gh CLI
    cmd = [
        "gh", "issue", "create",
        "--repo", "qut-dmrc/buttermilk",
        "--title", title,
        "--body", body,
        "--label", "enhancement"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Issue created successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create issue: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    create_issue()