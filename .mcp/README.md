# Buttermilk MCP Tools

This directory contains Model Context Protocol (MCP) tools designed to help AI assistants work more effectively on the Buttermilk project.

## Overview

These tools enforce the 9-step workflow from INSTRUCTIONS.md and provide structured access to common development tasks, reducing errors and improving consistency.

## Available Tools

### 1. `buttermilk-workflow-check`
Validates that the 9-step workflow is being followed correctly.

**Usage:**
```bash
buttermilk-workflow-check <step> [task_description]
```

**Steps:** STOP, ANALYZE, PLAN, TEST, IMPLEMENT, DOCUMENT, VALIDATE, COMMIT, REFLECT

**Example:**
```bash
buttermilk-workflow-check ANALYZE "Understanding WebSocket message flow"
```

### 2. `buttermilk-logs`
View and analyze Buttermilk server logs with various filtering options.

**Usage:**
```bash
buttermilk-logs <mode> [lines] [pattern]
```

**Modes:**
- `tail` - Show last N lines (default: 50)
- `errors` - Show only errors
- `warnings` - Show warnings and errors
- `search` - Search for pattern
- `websocket` - Show WebSocket messages
- `follow` - Follow log in real-time

**Example:**
```bash
buttermilk-logs errors 100
buttermilk-logs search 50 "AgentTrace"
```

### 3. `buttermilk-server`
Manage the Buttermilk API server.

**Usage:**
```bash
buttermilk-server <action> [debug] [flows]
```

**Actions:**
- `start` - Start the server
- `stop` - Stop the server
- `status` - Check server status
- `health` - Run health check
- `flows` - List available flows

**Example:**
```bash
buttermilk-server start true trans,zot,osb
buttermilk-server status
```

### 4. `buttermilk-test-flow`
Test specific Buttermilk flows with test data.

**Usage:**
```bash
buttermilk-test-flow <flow> <prompt> [mode]
```

**Modes:** `http` (default), `websocket`

**Example:**
```bash
buttermilk-test-flow zot "Find papers about AI ethics" http
```

### 5. `buttermilk-config-validate`
Validate YAML configurations and Hydra interpolations.

**Usage:**
```bash
buttermilk-config-validate <config_path> [check_interpolations]
```

**Example:**
```bash
buttermilk-config-validate conf/flows/zot.yaml true
```

### 6. `buttermilk-github-issue`
Search and manage GitHub issues for the project.

**Usage:**
```bash
buttermilk-github-issue <action> <query> [body] [labels]
```

**Actions:**
- `search` - Search for issues
- `create` - Create new issue
- `link` - Show how to link commits to issues

**Example:**
```bash
buttermilk-github-issue search "MCP tools"
buttermilk-github-issue create "Add logging feature" "Need better logging"
```

## Installation for AI Assistants

### For Claude (via MCP)

1. Add to your Claude configuration:
```json
{
  "mcpServers": {
    "buttermilk": {
      "command": "python",
      "args": ["/src/buttermilk/.mcp/buttermilk-mcp-server.py"],
      "cwd": "/src/buttermilk"
    }
  }
}
```

### For GitHub Copilot

Add to `.github/workflows/claude.yml`:
```yaml
custom_instructions: |
  Use the MCP tools in .mcp/tools/ for:
  - Workflow validation: .mcp/tools/buttermilk-workflow-check.sh
  - Log viewing: .mcp/tools/buttermilk-logs.sh
  - Server management: .mcp/tools/buttermilk-server.sh
  - Flow testing: .mcp/tools/buttermilk-test-flow.sh
  - Config validation: .mcp/tools/buttermilk-config-validate.sh
  - GitHub issues: .mcp/tools/buttermilk-github-issue.sh
```

### For VSCode/Cline

Create tasks in `.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Buttermilk: Check Workflow",
      "type": "shell",
      "command": "${workspaceFolder}/.mcp/tools/buttermilk-workflow-check.sh",
      "args": ["${input:workflowStep}", "${input:taskDescription}"],
      "problemMatcher": []
    },
    {
      "label": "Buttermilk: View Logs",
      "type": "shell",
      "command": "${workspaceFolder}/.mcp/tools/buttermilk-logs.sh",
      "args": ["tail", "50"],
      "problemMatcher": []
    }
  ],
  "inputs": [
    {
      "id": "workflowStep",
      "type": "pickString",
      "description": "Select workflow step",
      "options": ["STOP", "ANALYZE", "PLAN", "TEST", "IMPLEMENT", "DOCUMENT", "VALIDATE", "COMMIT", "REFLECT"]
    },
    {
      "id": "taskDescription",
      "type": "promptString",
      "description": "Enter task description"
    }
  ]
}
```

## Direct Usage

All tools can be run directly from the command line:

```bash
# Check you're following the workflow
.mcp/tools/buttermilk-workflow-check.sh PLAN

# View recent errors
.mcp/tools/buttermilk-logs.sh errors

# Start the server in debug mode
.mcp/tools/buttermilk-server.sh start true

# Test a flow
.mcp/tools/buttermilk-test-flow.sh zot "test query"

# Validate a config
.mcp/tools/buttermilk-config-validate.sh conf/flows/zot.yaml

# Search GitHub issues
.mcp/tools/buttermilk-github-issue.sh search "websocket"
```

## Benefits

1. **Consistency**: Enforces the 9-step workflow automatically
2. **Efficiency**: Reduces repetitive bash commands and manual checks
3. **Safety**: Validates changes before execution
4. **Guidance**: Helps AI assistants follow project standards
5. **Debugging**: Easier access to logs and diagnostics

## Contributing

To add new tools:

1. Add tool definition to `buttermilk-server.json`
2. Create implementation in `tools/` directory
3. Update `buttermilk-mcp-server.py` to handle the new tool
4. Update this README with usage instructions

Remember: All tools should follow the project's principles of reproducibility, traceability, and academic rigor.