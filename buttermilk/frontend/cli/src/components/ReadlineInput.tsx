import React, { useEffect, useRef } from 'react';
import { Box, Text } from 'ink';
import * as readline from 'readline';

interface Props {
  onSubmit: (text: string) => void;
  isActive: boolean;
}

const ReadlineInput = ({ onSubmit, isActive }: Props) => {
  const rlRef = useRef<readline.Interface | null>(null);

  useEffect(() => {
    if (!isActive) return;

    // Create readline interface
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: '❯ ',
      terminal: true
    });

    rlRef.current = rl;

    // Handle line input
    rl.on('line', (input) => {
      if (input.trim()) {
        onSubmit(input);
      }
      rl.prompt();
    });

    // Show initial prompt
    rl.prompt();

    // Cleanup
    return () => {
      rl.close();
    };
  }, [isActive, onSubmit]);

  // The actual input is handled by readline, this component just shows a placeholder
  return (
    <Box marginTop={1}>
      <Text dimColor>Ready for input...</Text>
    </Box>
  );
};

export default ReadlineInput;