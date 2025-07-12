import React, { useState, useEffect, useCallback } from 'react';
import { Box, Text, useApp, useStdin } from 'ink';
import * as readline from 'readline';
import { retroIRCTheme } from './themes.js';
import { Message } from './types.js';
import MessageList from './components/MessageList.js';
import Spinner from './components/Spinner.js';
import { connect, WebSocketConnection, ConnectionState } from './websocket.js';

interface Props {
  url: string;
}

const UIReadline = ({ url }: Props) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [connection, setConnection] = useState<WebSocketConnection | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const [connectionError, setConnectionError] = useState<Error | null>(null);
  const { exit } = useApp();
  const { stdin, setRawMode } = useStdin();

  useEffect(() => {
    const conn = connect(
      url,
      (message) => {
        setMessages((prevMessages) => [...prevMessages, message]);
      },
      (state, error) => {
        setConnectionState(state);
        setConnectionError(error || null);
      }
    );
    setConnection(conn);

    return () => {
      conn.close();
    };
  }, [url]);

  const handleSubmit = useCallback((text: string) => {
    if (!connection || !text.trim()) return;

    // Add local echo of user input
    const userMessage: Message = {
      type: 'user_message',
      payload: {
        message: text,
        timestamp: new Date().toISOString()
      }
    };
    setMessages(prev => [...prev, userMessage]);

    // Handle help command
    if (text === '/help') {
      const helpMessage: Message = {
        type: 'system_message',
        payload: {
          message: `Available commands:
  /flow <name> <prompt>  - Start a flow (e.g., /flow osb What is AI?)
  /run <name> <prompt>   - Alias for /flow
  /help                  - Show this help message
  /exit                  - Exit the CLI

You can also send raw JSON messages:
  {"type": "run_flow", "flow": "osb", "prompt": "Your question"}

Regular text is sent as user_message to the current flow.`
        }
      };
      setMessages(prev => [...prev, helpMessage]);
      return;
    }

    // Handle exit command
    if (text === '/exit') {
      connection.close();
      exit();
      return;
    }

    // Try to parse as JSON for advanced users
    if (text.trim().startsWith('{')) {
      try {
        const parsed = JSON.parse(text);
        connection.send(parsed);
        return;
      } catch (e) {
        // If JSON parsing fails, treat as regular message
      }
    }

    // Handle flow commands
    if (text.startsWith('/flow ') || text.startsWith('/run ')) {
      const parts = text.split(' ');
      const flowName = parts[1];
      const prompt = parts.slice(2).join(' ');
      
      if (!flowName) {
        setMessages(prev => [...prev, {
          type: 'system_error',
          payload: { message: 'Please specify a flow name. Usage: /flow <name> [prompt]' }
        }]);
        return;
      }

      const message: any = {
        type: 'run_flow',
        flow: flowName
      };
      if (prompt) {
        message.prompt = prompt;
      }
      connection.send(message);
      return;
    }

    // Default: send as manager_response (correct type for user input)
    connection.send({ type: 'manager_response', content: text } as any);
  }, [connection, exit]);

  useEffect(() => {
    if (connectionState !== 'connected' || !stdin) return;

    // Disable raw mode to allow readline to work properly
    setRawMode(false);

    const rl = readline.createInterface({
      input: stdin,
      output: process.stdout,
      prompt: '\n❯ ',
      terminal: true
    });

    rl.on('line', (input) => {
      // Clear the line to prevent duplicate display
      readline.moveCursor(process.stdout, 0, -1);
      readline.clearLine(process.stdout, 0);
      
      handleSubmit(input);
      
      // Show prompt again
      setTimeout(() => rl.prompt(), 100);
    });

    // Show initial prompt
    setTimeout(() => rl.prompt(), 100);

    return () => {
      rl.close();
      setRawMode(true);
    };
  }, [connectionState, stdin, setRawMode, handleSubmit]);

  const getStatusMessage = () => {
    const statusIcon = retroIRCTheme.format.status(
      connectionState === 'connected' ? 'connected' :
      connectionState === 'disconnected' || connectionState === 'error' ? 'disconnected' :
      'reconnecting'
    );
    
    const statusColor = 
      connectionState === 'connected' ? retroIRCTheme.colors.connected :
      connectionState === 'disconnected' || connectionState === 'error' ? retroIRCTheme.colors.disconnected :
      retroIRCTheme.colors.reconnecting;
    
    switch (connectionState) {
      case 'connecting':
        return <Text color={statusColor}><Spinner /> Connecting to server...</Text>;
      case 'connected':
        return <Text color={statusColor}>{statusIcon} Connected</Text>;
      case 'reconnecting':
        return <Text color={statusColor}><Spinner /> Reconnecting...</Text>;
      case 'disconnected':
        return <Text color={statusColor}>{statusIcon} Disconnected</Text>;
      case 'error':
        return <Text color={statusColor}>{statusIcon} Error: {connectionError?.message || 'Unknown error'}</Text>;
    }
  };

  return (
    <Box flexDirection="column">
      {/* Terminal header */}
      <Box 
        borderStyle="single" 
        borderColor={retroIRCTheme.colors.border}
        marginBottom={1}
      >
        <Box flexDirection="row" justifyContent="space-between" width="100%">
          <Text color={retroIRCTheme.colors.text} bold>
            Buttermilk Terminal v1.0 - IRC Mode
          </Text>
          <Box>
            {getStatusMessage()}
          </Box>
        </Box>
      </Box>
      
      {/* Message area */}
      <Box flexDirection="column" flexGrow={1}>
        <MessageList messages={messages} />
      </Box>
    </Box>
  );
};

export default UIReadline;