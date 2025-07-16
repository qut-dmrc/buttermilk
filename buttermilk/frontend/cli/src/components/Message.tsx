import React from 'react';
import { Box, Text } from 'ink';
import { retroIRCTheme } from '../themes.js';
import { 
  Message, 
  ChatMessage,
  FlowProgressUpdate,
  AgentAnnouncement,
  AgentOutput,
  OSBMessage,
  isChatMessage,
  isFlowEvent,
  isAgentMessage,
  isOSBMessage
} from '../types.js';

interface Props {
  message: Message;
}

// Extract agent name from various message types
const getAgentName = (message: Message): string => {
  const msg = message as any;
  
  // Check for agent_id in various locations
  if (msg.agent_id) return msg.agent_id;
  if (msg.agent) return msg.agent;
  if (msg.payload && msg.payload.agent_id) return msg.payload.agent_id;
  
  // Special handling for agent announcements - use the agent's name
  if (msg.type === 'agent_announcement' && msg.agent_id) {
    return msg.agent_id;
  }
  
  // Type-specific defaults
  switch (msg.type) {
    case 'user_message':
    case 'manager_response':
      return 'You';
    case 'system_message':
    case 'system_error':
      return 'System';
    case 'system_update':
      // Check if it's actually a TaskProcessingComplete that we should ignore
      if (msg.status === 'complete' || msg.task_status === 'complete') {
        return 'System';
      }
      return 'System';
    case 'flow_progress_update':
      return 'Flow';
    case 'agent_announcement':
      // This should have been handled above, but fallback
      return msg.agent_id || 'Agent';
    default:
      return msg.type.replace(/_/g, '-');
  }
};

// Get metadata string for agent (model, etc)
const getAgentMetadata = (message: Message): string | null => {
  if ('model' in message && (message as any).model) {
    return (message as any).model;
  }
  if (message.payload && typeof message.payload === 'object' && 'model' in message.payload) {
    return (message.payload as any).model;
  }
  if (message.type === 'flow_progress_update') {
    const flowMsg = message as FlowProgressUpdate;
    return flowMsg.status || null;
  }
  return null;
};

// Extract main content from message
const getMessageContent = (message: Message | any): string => {
  // Cast to any for flexibility since server messages might not match our types
  const msg = message as any;
  
  // Try all possible content locations
  // 1. Direct properties on the message
  if (msg.content && typeof msg.content === 'string') return msg.content;
  if (msg.message && typeof msg.message === 'string') return msg.message;
  if (msg.text && typeof msg.text === 'string') return msg.text;
  
  // 2. Properties in outputs (for agent messages)
  if (msg.outputs) {
    if (typeof msg.outputs === 'string') return msg.outputs;
    if (msg.outputs.content && typeof msg.outputs.content === 'string') return msg.outputs.content;
    if (msg.outputs.message && typeof msg.outputs.message === 'string') return msg.outputs.message;
    if (msg.outputs.text && typeof msg.outputs.text === 'string') return msg.outputs.text;
  }
  
  // 3. Properties in payload (for UI-created messages)
  if (msg.payload) {
    if (typeof msg.payload === 'string') return msg.payload;
    if (msg.payload.message && typeof msg.payload.message === 'string') return msg.payload.message;
    if (msg.payload.content && typeof msg.payload.content === 'string') return msg.payload.content;
    if (msg.payload.text && typeof msg.payload.text === 'string') return msg.payload.text;
  }
  
  // 4. Type-specific handling
  switch (msg.type) {
    case 'flow_progress_update':
      return msg.message || msg.step_name || msg.status || 'Flow update';
    
    case 'agent_announcement':
      return msg.status_message || `Agent ${msg.action || 'update'}`;
    
    case 'system_message':
    case 'system_update':
    case 'system_error':
      // These might have the content in various places
      return msg.message || msg.content || msg.text || 
             (msg.payload && (msg.payload.message || msg.payload.content)) ||
             `System: ${msg.type}`;
    
    case 'user_message':
    case 'manager_response':
      // User messages typically have content in payload.message
      return (msg.payload && msg.payload.message) || msg.message || msg.content || 'User message';
    
    case 'ui_message':
      // UI messages might have content in various locations
      return msg.outputs?.content || msg.content || msg.message || 
             (msg.payload && (msg.payload.message || msg.payload.content)) ||
             (msg.outputs?.options ? '⚠️ Action Required' : 'UI message');
    
    case 'chat_message':
      // Chat messages might have content in outputs
      return (msg.outputs && (msg.outputs.content || msg.outputs.message)) ||
             msg.content || msg.message || 'Chat message';
  }
  
  // 5. OSB message handling
  if (msg.type && msg.type.startsWith('osb_')) {
    if (msg.content) return msg.content;
    if (msg.policy_violations && Array.isArray(msg.policy_violations) && msg.policy_violations.length > 0) {
      return `Policy violations: ${msg.policy_violations.join(', ')}`;
    }
    if (msg.recommendations && Array.isArray(msg.recommendations) && msg.recommendations.length > 0) {
      return `Recommendations: ${msg.recommendations.join(', ')}`;
    }
  }
  
  // 6. Last resort - try to find any string property that looks like content
  const contentKeys = ['body', 'data', 'value', 'result', 'response', 'output'];
  for (const key of contentKeys) {
    if (msg[key] && typeof msg[key] === 'string') {
      return msg[key];
    }
  }
  
  // 7. If we have any object with reasonable size, stringify it
  const ignoredKeys = ['type', 'timestamp', 'agent_id', 'session_id', 'message_id'];
  const remainingKeys = Object.keys(msg).filter(k => !ignoredKeys.includes(k));
  if (remainingKeys.length > 0) {
    const contentObj: any = {};
    for (const key of remainingKeys) {
      contentObj[key] = msg[key];
    }
    const str = JSON.stringify(contentObj);
    if (str.length < 200 && str !== '{}') {
      return str;
    }
  }
  
  return `[${msg.type || 'unknown'}]`;
};

// Get color for message type
const getMessageColor = (message: Message): string => {
  const typeColors: Record<string, string> = {
    'user_message': retroIRCTheme.colors.info,
    'manager_response': retroIRCTheme.colors.info,
    'system_message': retroIRCTheme.colors.system,
    'system_update': retroIRCTheme.colors.warning,
    'system_error': retroIRCTheme.colors.error,
    'flow_progress_update': retroIRCTheme.colors.success,
    'agent_announcement': retroIRCTheme.colors.info,
    'ui_message': retroIRCTheme.colors.warning,
  };
  
  return typeColors[message.type] || retroIRCTheme.colors.text;
};

const MessageComponent = ({ message }: Props) => {
  const agentName = getAgentName(message);
  const metadata = getAgentMetadata(message);
  const content = getMessageContent(message);
  const timestamp = 'timestamp' in message && (message as any).timestamp 
    ? retroIRCTheme.format.timestamp((message as any).timestamp)
    : retroIRCTheme.format.timestamp(new Date());
  
  const agentColor = retroIRCTheme.agents.getColor(agentName);
  const messageColor = getMessageColor(message);
  
  // IRC-style layout: timestamp | nickname | message
  return (
    <Box marginBottom={0} flexDirection="row">
      {/* Timestamp */}
      <Text color={retroIRCTheme.colors.textDim}>
        {timestamp}
      </Text>
      
      {/* Separator */}
      <Text color={retroIRCTheme.colors.border}> </Text>
      
      {/* Nickname column (fixed width, right-aligned) */}
      <Box width={retroIRCTheme.layout.nickWidth} flexShrink={0}>
        <Text color={agentColor}>
          {retroIRCTheme.format.nickname(agentName)}
        </Text>
      </Box>
      
      {/* Separator */}
      <Text color={retroIRCTheme.colors.border}>
        {retroIRCTheme.layout.nickSeparator}
      </Text>
      
      {/* Message content */}
      <Box flexGrow={1} paddingLeft={1}>
        <Text color={messageColor} wrap="wrap">
          {content}
        </Text>
        
        {/* Special content rendering */}
        {message.type === 'agent_output' && (message as AgentOutput).tool_calls && (
          <Box flexDirection="column" paddingTop={0}>
            <Text color={retroIRCTheme.colors.warning} dimColor>
              Tool calls: {(message as AgentOutput).tool_calls?.length || 0}
            </Text>
          </Box>
        )}
        
        {/* UI Message with options (confirmation request) */}
        {message.type === 'ui_message' && (message as any).outputs?.options && (
          <Box flexDirection="column" paddingTop={1}>
            <Box borderStyle="single" borderColor={retroIRCTheme.colors.warning} paddingLeft={1} paddingRight={1}>
              <Box flexDirection="column">
                <Text color={retroIRCTheme.colors.warning} bold>
                  🔔 Confirmation Required:
                </Text>
                {Array.isArray((message as any).outputs.options) && (
                  <Box flexDirection="column" paddingTop={1}>
                    {(message as any).outputs.options.map((option: string, idx: number) => (
                      <Text key={idx} color={retroIRCTheme.colors.info}>
                        [{option[0].toUpperCase()}] {option}
                      </Text>
                    ))}
                  </Box>
                )}
                <Text color={retroIRCTheme.colors.textDim} dimColor italic>
                  Press ENTER to confirm, 'n' to reject
                </Text>
              </Box>
            </Box>
          </Box>
        )}

        {/* Debug info for UI messages */}
        {message.type === 'ui_message' && (
          <Box flexDirection="column" paddingTop={0}>
            <Text color={retroIRCTheme.colors.textDim} dimColor>
              [DEBUG] UIMessage - hasOptions: {!!(message as any).outputs?.options ? 'YES' : 'NO'}, 
              optionsType: {typeof (message as any).outputs?.options}, 
              isArray: {Array.isArray((message as any).outputs?.options) ? 'YES' : 'NO'}
            </Text>
          </Box>
        )}
        
        {isOSBMessage(message) && (
          <Box flexDirection="column" paddingTop={0}>
            {(message as OSBMessage).policy_violations && (message as OSBMessage).policy_violations!.length > 0 && (
              <Text color={retroIRCTheme.colors.error}>
                ⚠ Policy violations: {(message as OSBMessage).policy_violations!.length}
              </Text>
            )}
            {(message as OSBMessage).recommendations && (message as OSBMessage).recommendations!.length > 0 && (
              <Text color={retroIRCTheme.colors.success}>
                ✓ Recommendations: {(message as OSBMessage).recommendations!.length}
              </Text>
            )}
          </Box>
        )}
        
        {message.type === 'flow_progress_update' && (message as FlowProgressUpdate).waiting_on && (
          <Text color={retroIRCTheme.colors.warning} dimColor>
            Waiting on: {Object.keys((message as FlowProgressUpdate).waiting_on!).join(', ')}
          </Text>
        )}
      </Box>
    </Box>
  );
};

export default MessageComponent;