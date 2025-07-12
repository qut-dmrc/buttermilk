
import React, { useState } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';
import { retroIRCTheme } from '../themes.js';

interface Props {
  onSubmit: (text: string) => void;
}

const UserInput = ({ onSubmit }: Props) => {
  const [value, setValue] = useState('');
  
  const handleSubmit = (text: string) => {
    if (text.trim()) {
      onSubmit(text);
      setValue('');
    }
  };

  return (
    <Box>
      {/* IRC-style prompt with proper theming */}
      <Text color={retroIRCTheme.colors.text}>❯ </Text>
      <Box flexGrow={1}>
        <TextInput
          value={value}
          onChange={setValue}
          onSubmit={handleSubmit}
          placeholder="Type /help for commands..."
          showCursor={true}
          focus={true}
        />
      </Box>
    </Box>
  );
};

export default UserInput;
