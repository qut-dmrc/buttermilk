
import React, { useState } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';
import { retroIRCTheme } from '../themes.js';

interface Props {
  onSubmit: (text: string) => void;
}

const UserInput = ({ onSubmit }: Props) => {
  const [value, setValue] = useState('');
  
  const handleChange = (newValue: string) => {
    setValue(newValue);
  };
  
  const handleSubmit = (text: string) => {
    if (text.trim()) {
      onSubmit(text);
      setValue('');
    }
  };

  return (
    <Box>
      <Text color={retroIRCTheme.colors.text}>❯ </Text>
      <Box flexGrow={1}>
        <TextInput
          value={value}
          onChange={handleChange}
          onSubmit={handleSubmit}
          placeholder="Type /help for commands..."
          showCursor
        />
      </Box>
    </Box>
  );
};

export default UserInput;
