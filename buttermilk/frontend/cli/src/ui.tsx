import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { retroIRCTheme } from './themes.js';
import { Message } from './types.js';
import MessageList from './components/MessageList.js';
import UserInput from './components/UserInput.js';
import Spinner from './components/Spinner.js';
import { connect, WebSocketConnection, ConnectionState } from './websocket.js';

interface Props {
  url: string;
}

const UI = ({ url }: Props) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [connection, setConnection] = useState<WebSocketConnection | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const [connectionError, setConnectionError] = useState<Error | null>(null);
  const [lastUserInput, setLastUserInput] = useState<string>('');

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

  const handleSubmit = (text: string) => {
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
    setLastUserInput(text);

    // Handle help command
    if (text === '/help') {
      const helpMessage: Message = {
        type: 'system_message',
        payload: {
          message: `Available commands:
  /flow <name> <prompt>  - Start a flow (e.g., /flow osb What is AI?)
  /run <name> <prompt>   - Alias for /flow
  /help                  - Show this help message

You can also send raw JSON messages:
  {"type": "run_flow", "flow": "osb", "prompt": "Your question"}

Regular text is sent as user_message to the current flow.`
        }
      };
      setMessages(prev => [...prev, helpMessage]);
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
    // Note: manager_response expects fields directly, not wrapped in payload
    connection.send({ type: 'manager_response', content: text } as any);
  };

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
      
      {/* Input area - only show when connected */}
      {connectionState === 'connected' && (
        <Box borderStyle="single" borderColor={retroIRCTheme.colors.border} marginTop={1}>
          <UserInput onSubmit={handleSubmit} />
        </Box>
      )}
    </Box>
  );
};

export default UI;