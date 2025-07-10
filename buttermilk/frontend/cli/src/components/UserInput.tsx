
import React, { useState } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';

interface Props {
  onSubmit: (text: string) => void;
}

const UserInput = ({ onSubmit }: Props) => {
  const [value, setValue] = useState('');
  
  const handleSubmit = (text: string) => {
    setValue('');
    onSubmit(text);
  };

  return (
    <Box>
      <Text color="cyan">❯ </Text>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        placeholder="Type /help for commands..."
      />
    </Box>
  );
};

export default UserInput;
