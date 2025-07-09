# Buttermilk CLI Client

A beautiful command-line interface for interacting with Buttermilk flows via WebSocket.

## Features

- **WebSocket Communication**: Real-time bidirectional communication with Buttermilk backend
- **Automatic Reconnection**: Exponential backoff retry logic for robust connections
- **Rich Message Display**: Color-coded, formatted display for different message types
- **Connection Status**: Visual indicators for connection state
- **Configurable Backend**: Support for custom backend URLs via CLI args or environment variables

## Installation

```bash
npm install
npm run build
npm link  # Make available globally as buttermilk-cli
```

## Usage

### Basic Usage
```bash
buttermilk-cli
```

### Custom Backend
```bash
# Via command line arguments
buttermilk-cli --host myserver.com --port 9000
buttermilk-cli --url https://myserver.com:9000

# Via environment variables
export BUTTERMILK_HOST=myserver.com
export BUTTERMILK_PORT=9000
buttermilk-cli

# Or full URL
export BUTTERMILK_URL=https://myserver.com:9000
buttermilk-cli
```

## Message Types Supported

### Chat Messages
- User messages (cyan)
- System messages (gray)
- System updates (yellow)
- System errors (red)
- UI messages (magenta)
- Manager responses (cyan)
- Assessment results (blue)
- Research results (green)

### Flow Events
- Flow progress updates with status icons
- Agent announcements (join/leave/status)
- Task processing events
- Error events

### Agent Messages
- Agent outputs with content and tool calls
- Agent traces with full execution details
- Step requests and conductor decisions

### OSB Messages
- Multi-agent analysis from:
  - Researcher (blue)
  - Policy Analyst (magenta)
  - Fact Checker (yellow)
  - Explorer (cyan)
  - Synthesizer (green)
- Policy violations and recommendations

## Connection States

- 🟡 **Connecting**: Initial connection attempt
- 🟢 **Connected**: Active WebSocket connection
- 🟡 **Reconnecting**: Attempting to reconnect after disconnect
- 🔴 **Disconnected**: Connection closed
- 🔴 **Error**: Connection error occurred

## Development

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Run locally
node dist/cli.js
```

## Architecture

- **cli.tsx**: Main entry point, handles session initialization
- **ui.tsx**: Main UI component with connection management
- **websocket.ts**: Enhanced WebSocket client with reconnection logic
- **types.ts**: Comprehensive message type definitions
- **components/Message.tsx**: Rich message rendering with type-specific formatting
- **components/UserInput.tsx**: User input handling
- **components/Spinner.tsx**: Loading indicator