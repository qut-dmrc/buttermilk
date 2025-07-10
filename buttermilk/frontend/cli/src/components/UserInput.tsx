
import React, { useState } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';

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
      <Text color="cyan">❯ </Text>
      <TextInput
        value={value}
        onChange={handleChange}
        onSubmit={handleSubmit}
        placeholder="Type /help for commands..."
        showCursor
      />
    </Box>
  );
};

export default UserInput;
