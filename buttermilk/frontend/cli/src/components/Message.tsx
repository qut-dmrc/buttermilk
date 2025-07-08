
import React from 'react';
import { Box, Text } from 'ink';
import { Message } from '../types.js';

interface Props {
  message: Message;
}

const MessageComponent = ({ message }: Props) => {
  return (
    <Box borderStyle="round" padding={1} flexDirection="column">
      <Text color="blue">{message.type}</Text>
      <Text>{JSON.stringify(message.payload, null, 2)}</Text>
    </Box>
  );
};

export default MessageComponent;
