import React from 'react';
import { Box, Text } from 'ink';
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

const formatTimestamp = (timestamp?: string) => {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  return date.toLocaleTimeString();
};

const MessageComponent = ({ message }: Props) => {
  const renderChatMessage = (msg: ChatMessage) => {
    const typeColors: Record<string, string> = {
      'user_message': 'cyan',
      'system_message': 'gray',
      'system_update': 'yellow',
      'system_error': 'red',
      'ui_message': 'magenta',
      'manager_response': 'cyan',
      'assessments': 'blue',
      'research_result': 'green',
      'differences': 'yellow',
      'judge_reasons': 'blue'
    };

    const color = typeColors[msg.type] || 'white';
    const icon = msg.type === 'user_message' ? '👤' : 
                 msg.type === 'system_error' ? '❌' :
                 msg.type === 'system_update' ? '📢' : '💬';

    return (
      <Box flexDirection="column" marginBottom={1}>
        <Box>
          <Text color={color} bold>
            {icon} {msg.type.replace(/_/g, ' ').toUpperCase()}
          </Text>
          {msg.timestamp && (
            <Text color="gray" dimColor> [{formatTimestamp(msg.timestamp)}]</Text>
          )}
        </Box>
        {msg.preview && (
          <Box paddingLeft={2}>
            <Text dimColor>{msg.preview}</Text>
          </Box>
        )}
        <Box paddingLeft={2}>
          <Text>{JSON.stringify(msg.payload || msg.outputs, null, 2)}</Text>
        </Box>
        {msg.tracing_link && (
          <Box paddingLeft={2}>
            <Text color="blue" dimColor>🔗 {msg.tracing_link}</Text>
          </Box>
        )}
      </Box>
    );
  };

  const renderFlowEvent = (msg: FlowProgressUpdate | AgentAnnouncement) => {
    if (msg.type === 'flow_progress_update') {
      const statusColors = {
        'STARTED': 'green',
        'IN_PROGRESS': 'yellow',
        'COMPLETED': 'green',
        'ERROR': 'red'
      };
      const statusIcons = {
        'STARTED': '▶️',
        'IN_PROGRESS': '⏳',
        'COMPLETED': '✅',
        'ERROR': '❌'
      };

      const status = msg.status || 'IN_PROGRESS';
      const color = statusColors[status] || 'white';
      const icon = statusIcons[status] || '●';

      return (
        <Box flexDirection="column" marginY={0.5}>
          <Box>
            <Text color={color}>
              {icon} Flow Progress: {msg.step_name || 'Unknown Step'}
            </Text>
          </Box>
          {msg.message && (
            <Box paddingLeft={2}>
              <Text dimColor>{msg.message}</Text>
            </Box>
          )}
          {msg.waiting_on && Object.keys(msg.waiting_on).length > 0 && (
            <Box paddingLeft={2}>
              <Text color="yellow" dimColor>
                Waiting on: {Object.entries(msg.waiting_on).map(([agent, task]) => `${agent}: ${task}`).join(', ')}
              </Text>
            </Box>
          )}
        </Box>
      );
    }

    if (msg.type === 'agent_announcement') {
      const actionIcons = {
        'joined': '➕',
        'left': '➖',
        'status': '📊'
      };
      const icon = actionIcons[msg.action || 'status'] || '●';

      return (
        <Box>
          <Text color="blue">
            {icon} Agent {msg.agent_id}: {msg.action || 'status update'}
          </Text>
          {msg.status_message && (
            <Text color="gray"> - {msg.status_message}</Text>
          )}
        </Box>
      );
    }

    return renderGenericMessage(msg);
  };

  const renderAgentMessage = (msg: AgentOutput) => {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <Box>
          <Text color="green" bold>
            🤖 Agent Output
          </Text>
          {msg.agent_id && (
            <Text color="gray" dimColor> [{msg.agent_id}]</Text>
          )}
        </Box>
        {msg.content && (
          <Box paddingLeft={2} flexDirection="column">
            <Text>{msg.content}</Text>
          </Box>
        )}
        {msg.tool_calls && msg.tool_calls.length > 0 && (
          <Box paddingLeft={2} flexDirection="column">
            <Text color="yellow">Tool Calls:</Text>
            {msg.tool_calls.map((tool, i) => (
              <Text key={i} color="gray" dimColor>
                • {JSON.stringify(tool)}
              </Text>
            ))}
          </Box>
        )}
        {msg.payload && (
          <Box paddingLeft={2}>
            <Text dimColor>{JSON.stringify(msg.payload, null, 2)}</Text>
          </Box>
        )}
      </Box>
    );
  };

  const renderOSBMessage = (msg: OSBMessage) => {
    const agentColors = {
      'researcher': 'blue',
      'policy_analyst': 'magenta',
      'fact_checker': 'yellow',
      'explorer': 'cyan',
      'synthesizer': 'green'
    };
    const agentIcons = {
      'researcher': '🔬',
      'policy_analyst': '📋',
      'fact_checker': '✓',
      'explorer': '🔍',
      'synthesizer': '🔗'
    };

    const color = agentColors[msg.agent || 'synthesizer'] || 'white';
    const icon = agentIcons[msg.agent || 'synthesizer'] || '●';

    return (
      <Box flexDirection="column" marginBottom={1}>
        <Box>
          <Text color={color} bold>
            {icon} OSB {msg.agent ? msg.agent.replace(/_/g, ' ').toUpperCase() : 'MESSAGE'}: {msg.type.replace('osb_', '').toUpperCase()}
          </Text>
        </Box>
        {msg.content && (
          <Box paddingLeft={2}>
            <Text>{msg.content}</Text>
          </Box>
        )}
        {msg.policy_violations && msg.policy_violations.length > 0 && (
          <Box paddingLeft={2} flexDirection="column">
            <Text color="red" bold>⚠️  Policy Violations:</Text>
            {msg.policy_violations.map((violation, i) => (
              <Text key={i} color="red">• {JSON.stringify(violation)}</Text>
            ))}
          </Box>
        )}
        {msg.recommendations && msg.recommendations.length > 0 && (
          <Box paddingLeft={2} flexDirection="column">
            <Text color="green" bold>💡 Recommendations:</Text>
            {msg.recommendations.map((rec, i) => (
              <Text key={i} color="green">• {JSON.stringify(rec)}</Text>
            ))}
          </Box>
        )}
        {msg.payload && (
          <Box paddingLeft={2}>
            <Text dimColor>{JSON.stringify(msg.payload, null, 2)}</Text>
          </Box>
        )}
      </Box>
    );
  };

  const renderGenericMessage = (msg: Message) => {
    return (
      <Box borderStyle="round" padding={1} flexDirection="column">
        <Text color="blue">{msg.type}</Text>
        <Text>{JSON.stringify(msg.payload, null, 2)}</Text>
      </Box>
    );
  };

  // Route to appropriate renderer based on message type
  if (isChatMessage(message)) {
    return renderChatMessage(message as ChatMessage);
  } else if (isFlowEvent(message)) {
    return renderFlowEvent(message as FlowProgressUpdate | AgentAnnouncement);
  } else if (isAgentMessage(message)) {
    return renderAgentMessage(message as AgentOutput);
  } else if (isOSBMessage(message)) {
    return renderOSBMessage(message as OSBMessage);
  } else {
    return renderGenericMessage(message);
  }
};

export default MessageComponent;