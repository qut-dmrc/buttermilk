
import React, { useState } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';

interface Props {
  onSubmit: (text: string) => void;
}

const UserInput = ({ onSubmit }: Props) => {
  const [value, setValue] = useState('');

  return (
    <Box>
      <Box marginRight={1}>
        <Text>You:</Text>
      </Box>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={onSubmit}
      />
    </Box>
  );
};

export default UserInput;
