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
    if (connection) {
      connection.send({ type: 'user_message', payload: { text } });
    }
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