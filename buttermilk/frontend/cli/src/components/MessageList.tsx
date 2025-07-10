import React from 'react';
import { Box, Text } from 'ink';
import { retroIRCTheme } from '../themes.js';
import { Message } from '../types.js';
import MessageComponent from './Message.js';

interface Props {
  messages: Message[];
}

// Helper to check if we should filter out a message
const shouldFilterMessage = (message: Message): boolean => {
  const msg = message as any;
  
  // Filter out TaskProcessingComplete messages that appear as system_update
  if (msg.type === 'system_update' && 
      (msg.status === 'complete' || 
       msg.task_status === 'complete' ||
       msg.message?.includes('TaskProcessingComplete'))) {
    return true;
  }
  
  return false;
};

const MessageList = ({ messages }: Props) => {
  // Filter out messages we don't want to display
  const filteredMessages = messages.filter(msg => !shouldFilterMessage(msg));
  
  // Show a welcome message if no messages yet
  if (filteredMessages.length === 0) {
    return (
      <Box flexDirection="column" paddingTop={1}>
        <Text color={retroIRCTheme.colors.textDim}>
          {retroIRCTheme.format.timestamp(new Date())} {retroIRCTheme.format.nickname('System', 15)}{retroIRCTheme.layout.nickSeparator} Welcome to Buttermilk Terminal
        </Text>
        <Text color={retroIRCTheme.colors.textDim}>
          {retroIRCTheme.format.timestamp(new Date())} {retroIRCTheme.format.nickname('System', 15)}{retroIRCTheme.layout.nickSeparator} Type /help for available commands
        </Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      {filteredMessages.map((msg, i) => (
        <MessageComponent key={i} message={msg} />
      ))}
    </Box>
  );
};

export default MessageList;