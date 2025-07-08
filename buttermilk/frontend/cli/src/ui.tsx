
import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { Message } from './types.js';
import MessageComponent from './components/Message.js';
import UserInput from './components/UserInput.js';
import Spinner from './components/Spinner.js';
import { connect } from './websocket.js';

interface Props {
  url: string;
}

const UI = ({ url }: Props) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [connection, setConnection] = useState<{ send: (msg: Message) => void; close: () => void; } | null>(null);

  useEffect(() => {
    const conn = connect(url, (message) => {
      setMessages((prevMessages) => [...prevMessages, message]);
    });
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

  return (
    <Box flexDirection="column">
      {
        messages.map((msg, i) => (
          <MessageComponent key={i} message={msg} />
        ))
      }
      { !connection && <Spinner /> }
      { connection && <UserInput onSubmit={handleSubmit} /> }
    </Box>
  );
};

export default UI;
