import React from 'react';
import { Box, Text } from 'ink';
import { retroIRCTheme } from '../themes.js';

interface ProgressIndicatorProps {
  activeAgents: number;
  currentStep?: string;
  status?: string;
  waitingOn?: string[];
}

const ProgressIndicator = ({ activeAgents, currentStep, status, waitingOn }: ProgressIndicatorProps) => {
  if (activeAgents === 0 && !currentStep) {
    return null; // Don't show indicator when nothing is happening
  }

  const getStatusColor = (status?: string) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return retroIRCTheme.colors.success;
      case 'error':
        return retroIRCTheme.colors.error;
      case 'started':
      case 'in_progress':
        return retroIRCTheme.colors.warning;
      default:
        return retroIRCTheme.colors.info;
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return '✓';
      case 'error':
        return '✗';
      case 'started':
      case 'in_progress':
        return '⟳';
      default:
        return '•';
    }
  };

  return (
    <Box borderStyle="round" borderColor={retroIRCTheme.colors.border} paddingX={1} marginBottom={1}>
      <Text color={retroIRCTheme.colors.textDim}>
        {getStatusIcon(status)} Flow: {currentStep || 'Running'}
      </Text>
      {activeAgents > 0 && (
        <Text color={retroIRCTheme.colors.info}>
          {' '} | {activeAgents} agent{activeAgents !== 1 ? 's' : ''} active
        </Text>
      )}
      {waitingOn && waitingOn.length > 0 && (
        <Text color={retroIRCTheme.colors.warning}>
          {' '} | Waiting on: {waitingOn.join(', ')}
        </Text>
      )}
    </Box>
  );
};

export default ProgressIndicator;