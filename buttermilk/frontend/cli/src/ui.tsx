import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { Message } from './types.js';
import MessageComponent from './components/Message.js';
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

    // Default: send as ui_message (supported by backend)
    connection.send({ type: 'ui_message', payload: { text } });
  };

  const getStatusMessage = () => {
    switch (connectionState) {
      case 'connecting':
        return <Text color="yellow"><Spinner /> Connecting to server...</Text>;
      case 'connected':
        return <Text color="green">● Connected</Text>;
      case 'reconnecting':
        return <Text color="yellow"><Spinner /> Reconnecting...</Text>;
      case 'disconnected':
        return <Text color="red">● Disconnected</Text>;
      case 'error':
        return <Text color="red">● Error: {connectionError?.message || 'Unknown error'}</Text>;
    }
  };

  return (
    <Box flexDirection="column">
      <Box marginBottom={1} borderStyle="single" borderColor="gray">
        <Box padding={0.5}>
          {getStatusMessage()}
        </Box>
      </Box>
      
      <Box flexDirection="column" flexGrow={1}>
        {messages.map((msg, i) => (
          <MessageComponent key={i} message={msg} />
        ))}
      </Box>
      
      {connectionState === 'connected' && <UserInput onSubmit={handleSubmit} />}
    </Box>
  );
};

export default UI;