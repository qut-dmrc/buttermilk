# Buttermilk CLI Development Guide

## Overview

The Buttermilk CLI is a React-based terminal interface built with Ink that provides real-time interaction with Buttermilk flows via WebSocket.

## Architecture

- **React + Ink**: Terminal UI framework for interactive CLI
- **WebSocket**: Real-time bidirectional communication
- **TypeScript**: Type-safe development
- **Modular Components**: Separated concerns for messages, input, connection

## Key Features Implemented

### 1. WebSocket Communication
- Automatic session ID retrieval
- Connection with exponential backoff retry
- Message queueing for offline sends
- Connection status indicators

### 2. Rich Message Display
- Color-coded message types
- Icons for visual clarity
- Support for all Buttermilk message types:
  - Chat messages (user, system, errors)
  - Flow events (progress, agent announcements)
  - Agent messages (outputs, traces)
  - OSB-specific messages
  - Interactive UI messages

### 3. Flow Commands
- `/flow <name> [prompt]` - Start a flow
- `/run <name> [prompt]` - Alias for /flow
- `/help` - Show available commands
- Raw JSON message support for advanced users

### 4. Configuration
- CLI arguments: `--host`, `--port`, `--url`, `--debug`
- Environment variables: `BUTTERMILK_HOST`, `BUTTERMILK_PORT`, `BUTTERMILK_URL`, `BUTTERMILK_DEBUG`
- Support for HTTP/HTTPS and WS/WSS protocols

### 5. Debug Mode
- Detailed logging with timestamps
- WebSocket message tracing
- Connection state tracking

## Testing Framework

### Mock Server Tests
```bash
npm test          # Run with mock servers
npm test:debug    # With debug output
```

### Real Server Tests
```bash
npm run test:real          # Basic output
npm run test:real:debug    # Full debug output
npm run test:real:verbose  # Key messages from passed tests
```

### Test Components
1. **MockWebSocketServer**: Simulates backend for isolated testing
2. **TestClient**: Automated CLI testing with input simulation
3. **ScenarioRunner**: Executes test scenarios with assertions
4. **Real Server Tests**: Integration tests against live backend

## Message Format

The backend expects messages at the top level, not nested in payload:

```javascript
// Correct format for run_flow
{
  "type": "run_flow",
  "flow": "osb",
  "prompt": "What is AI?"
}

// User messages
{
  "type": "user_message",
  "payload": { "text": "Hello" }
}

// Manager responses (for interactive flows)
{
  "type": "manager_response",
  "payload": { "text": "yes" }
}
```

## Development Workflow

### Running in Development
```bash
# Build and watch for changes
npm run dev

# In another terminal, run the CLI
node dist/cli.js --host localhost --port 8000 --debug
```

### Testing a Flow
1. Start the Buttermilk backend
2. Run the CLI: `node dist/cli.js`
3. Once connected, use `/flow test` or `/flow osb Your question here`
4. Respond to any prompts (type `yes` or `no` for confirmations)

### Debugging Connection Issues
1. Use `--debug` flag for detailed logging
2. Check backend is running on correct port
3. Verify session endpoint returns valid session ID
4. Check WebSocket endpoint accepts connections

## Known Issues

1. Raw mode not supported when piping input (Ink limitation)
2. WebSocket may close if backend doesn't send messages
3. Some flows may require specific backend configuration

## Future Enhancements

- Command history (up/down arrows)
- Tab completion for flow names
- Session persistence/resume
- Multiple concurrent flows
- File upload support
- Export conversation history