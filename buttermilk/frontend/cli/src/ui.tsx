import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { retroIRCTheme } from './themes.js';
import { Message } from './types.js';
import MessageList from './components/MessageList.js';
import UserInput from './components/UserInput.js';
import Spinner from './components/Spinner.js';
import ProgressIndicator from './components/ProgressIndicator.js';
import { connect, WebSocketConnection, ConnectionState } from './websocket.js';

interface Props {
  url: string;
}

const UI = ({ url }: Props) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [connection, setConnection] = useState<WebSocketConnection | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const [connectionError, setConnectionError] = useState<Error | null>(null);
  const [lastUserInput, setLastUserInput] = useState<string>('');

  // Progress tracking state
  const [activeAgents, setActiveAgents] = useState<Set<string>>(new Set());
  const [currentStep, setCurrentStep] = useState<string>('');
  const [flowStatus, setFlowStatus] = useState<string>('');
  const [waitingOn, setWaitingOn] = useState<string[]>([]);

  // Message types to filter out from chat display
  const FILTERED_MESSAGE_TYPES = new Set([
    'flow_progress_update',
    'agent_announcement',
    'task_processing_started',
    'task_processing_complete'
  ]);

  useEffect(() => {
    const conn = connect(
      url,
      (message) => {
        // Handle progress updates separately
        if (message.type === 'flow_progress_update') {
          updateFlowProgress(message);
        } else if (message.type === 'agent_announcement') {
          updateAgentStatus(message);
        } else if (message.type === 'task_processing_started') {
          // Track when agents start processing
          if (message.payload?.agent_id) {
            setActiveAgents(prev => new Set([...prev, message.payload.agent_id]));
          }
        } else if (message.type === 'task_processing_complete') {
          // Track when agents finish processing
          if (message.payload?.agent_id) {
            setActiveAgents(prev => {
              const newSet = new Set(prev);
              newSet.delete(message.payload.agent_id);
              return newSet;
            });
          }
        }

        // Only add non-filtered messages to the chat display
        if (!FILTERED_MESSAGE_TYPES.has(message.type)) {
          setMessages((prevMessages) => [...prevMessages, message]);
        }
      },
      (state, error) => {
        setConnectionState(state);
        setConnectionError(error || null);
      }
    );
    setConnection(conn);

    return () => {
      conn.close();
    };
  }, [url]);

  const updateFlowProgress = (message: Message) => {
    const payload = message.payload;
    if (payload?.step_name) {
      setCurrentStep(payload.step_name);
    }
    if (payload?.status) {
      setFlowStatus(payload.status);
    }
    if (payload?.waiting_on && typeof payload.waiting_on === 'object') {
      setWaitingOn(Object.keys(payload.waiting_on));
    }
  };

  const updateAgentStatus = (message: Message) => {
    const payload = message.payload;
    if (payload?.agent_id) {
      if (payload.action === 'joined') {
        setActiveAgents(prev => new Set([...prev, payload.agent_id]));
      } else if (payload.action === 'left') {
        setActiveAgents(prev => {
          const newSet = new Set(prev);
          newSet.delete(payload.agent_id);
          return newSet;
        });
      }
    }
  };

  const handleSubmit = (text: string) => {
    if (!connection || !text.trim()) return;

    // Add local echo of user input
    const userMessage: Message = {
      type: 'user_message',
      payload: {
        message: text,
        timestamp: new Date().toISOString()
      }
    };
    setMessages(prev => [...prev, userMessage]);
    setLastUserInput(text);

    // Handle help command
    if (text === '/help') {
      const helpMessage: Message = {
        type: 'system_message',
        payload: {
          message: `Available commands:
  /flow <name> <prompt>  - Start a flow (e.g., /flow osb What is AI?)
  /run <name> <prompt>   - Alias for /flow
  /help                  - Show this help message

You can also send raw JSON messages:
  {"type": "run_flow", "flow": "osb", "prompt": "Your question"}

Regular text is sent as user_message to the current flow.`
        }
      };
      setMessages(prev => [...prev, helpMessage]);
      return;
    }

    // Try to parse as JSON for advanced users
    if (text.trim().startsWith('{')) {
      try {
        const parsed = JSON.parse(text);
        connection.send(parsed);
        return;
      } catch (e) {
        // If JSON parsing fails, treat as regular message
      }
    }

    // Handle flow commands
    if (text.startsWith('/flow ') || text.startsWith('/run ')) {
      const parts = text.split(' ');
      const flowName = parts[1];
      const prompt = parts.slice(2).join(' ');
      
      if (!flowName) {
        setMessages(prev => [...prev, {
          type: 'system_error',
          payload: { message: 'Please specify a flow name. Usage: /flow <name> [prompt]' }
        }]);
        return;
      }

      const message: any = {
        type: 'run_flow',
        flow: flowName
      };
      if (prompt) {
        message.prompt = prompt;
      }
      connection.send(message);
      return;
    }

    // Default: send as manager_response (correct type for user input)
    // Note: manager_response expects fields directly, not wrapped in payload
    connection.send({ type: 'manager_response', content: text } as any);
  };

  const getStatusMessage = () => {
    const statusIcon = retroIRCTheme.format.status(
      connectionState === 'connected' ? 'connected' :
      connectionState === 'disconnected' || connectionState === 'error' ? 'disconnected' :
      'reconnecting'
    );
    
    const statusColor = 
      connectionState === 'connected' ? retroIRCTheme.colors.connected :
      connectionState === 'disconnected' || connectionState === 'error' ? retroIRCTheme.colors.disconnected :
      retroIRCTheme.colors.reconnecting;
    
    switch (connectionState) {
      case 'connecting':
        return <Text color={statusColor}><Spinner /> Connecting to server...</Text>;
      case 'connected':
        return <Text color={statusColor}>{statusIcon} Connected</Text>;
      case 'reconnecting':
        return <Text color={statusColor}><Spinner /> Reconnecting...</Text>;
      case 'disconnected':
        return <Text color={statusColor}>{statusIcon} Disconnected</Text>;
      case 'error':
        return <Text color={statusColor}>{statusIcon} Error: {connectionError?.message || 'Unknown error'}</Text>;
    }
  };

  return (
    <Box flexDirection="column">
      {/* Terminal header */}
      <Box 
        borderStyle="single" 
        borderColor={retroIRCTheme.colors.border}
        marginBottom={1}
      >
        <Box flexDirection="row" justifyContent="space-between" width="100%">
          <Text color={retroIRCTheme.colors.text} bold>
            Buttermilk Terminal v1.0 - IRC Mode
          </Text>
          <Box>
            {getStatusMessage()}
          </Box>
        </Box>
      </Box>
      
      {/* Progress indicator */}
      <ProgressIndicator 
        activeAgents={activeAgents.size}
        currentStep={currentStep}
        status={flowStatus}
        waitingOn={waitingOn}
      />
      
      {/* Message area */}
      <Box flexDirection="column" flexGrow={1}>
        <MessageList messages={messages} />
      </Box>
      
      {/* Input area - only show when connected */}
      {connectionState === 'connected' && (
        <Box borderStyle="single" borderColor={retroIRCTheme.colors.border} marginTop={1}>
          <UserInput onSubmit={handleSubmit} />
        </Box>
      )}
    </Box>
  );
};

export default UI;