# MCP Tools Integration Examples

This document shows how to integrate Buttermilk MCP tools with various AI assistants.

## Claude (via Claude Desktop)

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "buttermilk": {
      "command": "python3",
      "args": ["/src/buttermilk/.mcp/buttermilk-mcp-server.py", "serve"],
      "env": {
        "PYTHONPATH": "/src/buttermilk"
      }
    }
  }
}
```

## GitHub Actions (for Claude Code)

Update `.github/workflows/claude.yml`:

```yaml
- name: Run Claude Code
  uses: anthropics/claude-code-action@beta
  with:
    custom_instructions: |
      CRITICAL: Follow instructions in @docs/bots/INSTRUCTIONS.md
      
      Use these MCP tools for development:
      - Workflow check: .mcp/tools/buttermilk-workflow-check.sh <step>
      - View logs: .mcp/tools/buttermilk-logs.sh <mode>
      - Server control: .mcp/tools/buttermilk-server.sh <action>
      - Test flows: .mcp/tools/buttermilk-test-flow.sh <flow> <prompt>
      - Validate config: .mcp/tools/buttermilk-config-validate.sh <path>
      - GitHub issues: .mcp/tools/buttermilk-github-issue.sh <action> <query>
      
      ALWAYS use buttermilk-workflow-check before making changes!
```

## VSCode Settings (for Copilot/Cline)

Add to `.vscode/settings.json`:

```json
{
  "buttermilk.mcp.enabled": true,
  "buttermilk.mcp.tools": {
    "workflowCheck": "${workspaceFolder}/.mcp/tools/buttermilk-workflow-check.sh",
    "logs": "${workspaceFolder}/.mcp/tools/buttermilk-logs.sh",
    "server": "${workspaceFolder}/.mcp/tools/buttermilk-server.sh",
    "testFlow": "${workspaceFolder}/.mcp/tools/buttermilk-test-flow.sh",
    "configValidate": "${workspaceFolder}/.mcp/tools/buttermilk-config-validate.sh",
    "githubIssue": "${workspaceFolder}/.mcp/tools/buttermilk-github-issue.sh"
  }
}
```

## Shell Aliases (for manual use)

Add to your shell configuration (`.bashrc`, `.zshrc`, etc.):

```bash
# Buttermilk MCP tool aliases
alias bm-check='/src/buttermilk/.mcp/tools/buttermilk-workflow-check.sh'
alias bm-logs='/src/buttermilk/.mcp/tools/buttermilk-logs.sh'
alias bm-server='/src/buttermilk/.mcp/tools/buttermilk-server.sh'
alias bm-test='/src/buttermilk/.mcp/tools/buttermilk-test-flow.sh'
alias bm-config='/src/buttermilk/.mcp/tools/buttermilk-config-validate.sh'
alias bm-issue='/src/buttermilk/.mcp/tools/buttermilk-github-issue.sh'

# Quick workflow status
bm-status() {
    echo "Current workflow step:"
    bm-check "${1:-STOP}"
}
```

## Example Workflow

Here's how an AI assistant should use these tools:

```bash
# 1. Start by checking workflow
bm-check STOP "Implementing new logging feature"

# 2. Search for existing issues
bm-issue search "logging"

# 3. Analyze the system
bm-check ANALYZE
bm-logs errors  # Check current errors

# 4. Create plan
bm-check PLAN
bm-issue create "Add structured logging" "Need to implement structured logging..."

# 5. Write tests first
bm-check TEST

# 6. Implement
bm-check IMPLEMENT
bm-server start true  # Start in debug mode
bm-test zot "test query"  # Test the flow

# 7. Document
bm-check DOCUMENT

# 8. Validate
bm-check VALIDATE
bm-config conf/flows/zot.yaml

# 9. Commit
bm-check COMMIT

# 10. Reflect
bm-check REFLECT
```

## Custom Instructions Template

For AI assistants, use this template:

```
When working on Buttermilk:
1. ALWAYS start with: buttermilk-workflow-check STOP
2. Follow the 9-step workflow WITHOUT EXCEPTIONS
3. Use buttermilk-logs to debug issues
4. Use buttermilk-server to manage the API
5. Use buttermilk-test-flow to test changes
6. Use buttermilk-config-validate before committing
7. Use buttermilk-github-issue to track work

The workflow steps are:
STOP → ANALYZE → PLAN → TEST → IMPLEMENT → DOCUMENT → VALIDATE → COMMIT → REFLECT
```

## Troubleshooting

If tools aren't working:

1. Check permissions: `ls -la .mcp/tools/`
2. Make executable: `chmod +x .mcp/tools/*.sh`
3. Test individually: `.mcp/test-tools.sh`
4. Check logs: `buttermilk-logs errors`

Remember: These tools enforce best practices - use them consistently!