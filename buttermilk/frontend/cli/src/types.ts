export interface Message {
  type: string;
  payload: any;
}

// Base message types from Buttermilk
export interface ChatMessage extends Message {
  type: 'chat_message' | 'record' | 'ui_message' | 'manager_response' | 
        'system_message' | 'system_update' | 'system_error' | 'user_message' |
        'assessments' | 'research_result' | 'differences' | 'judge_reasons';
  message_id?: string;
  preview?: string;
  outputs?: any;
  timestamp?: string;
  agent_info?: any;
  tracing_link?: string;
}

// Flow event types
export interface FlowEvent extends Message {
  type: 'flow_event' | 'error_event' | 'flow_progress_update' | 
        'task_processing_started' | 'task_processing_complete' | 'agent_announcement';
  source?: string;
  timestamp?: string;
}

export interface FlowProgressUpdate extends FlowEvent {
  type: 'flow_progress_update';
  step_name?: string;
  status?: 'STARTED' | 'IN_PROGRESS' | 'COMPLETED' | 'ERROR';
  message?: string;
  waiting_on?: Record<string, string>;
}

export interface AgentAnnouncement extends FlowEvent {
  type: 'agent_announcement';
  agent_id?: string;
  action?: 'joined' | 'left' | 'status';
  status_message?: string;
}

// Agent communication messages
export interface AgentMessage extends Message {
  type: 'agent_input' | 'agent_output' | 'agent_trace' | 'step_request' | 
        'conductor_request' | 'tool_output';
  agent_id?: string;
  session_id?: string;
  trace_id?: string;
}

export interface AgentOutput extends AgentMessage {
  type: 'agent_output';
  content?: string;
  tool_calls?: any[];
  metadata?: Record<string, any>;
}

// OSB-specific messages
export interface OSBMessage extends Message {
  type: 'osb_query' | 'osb_status' | 'osb_partial' | 'osb_complete' | 'osb_error';
  agent?: 'researcher' | 'policy_analyst' | 'fact_checker' | 'explorer' | 'synthesizer';
  content?: string;
  analysis?: any;
  policy_violations?: any[];
  recommendations?: any[];
}

// WebSocket control messages
export interface ControlMessage extends Message {
  type: 'run_flow' | 'pull_task' | 'pull_tox';
  flow_id?: string;
  task_id?: string;
  parameters?: Record<string, any>;
}

// Type guards
export const isChatMessage = (msg: Message): msg is ChatMessage => {
  return ['chat_message', 'record', 'ui_message', 'manager_response', 
          'system_message', 'system_update', 'system_error', 'user_message',
          'assessments', 'research_result', 'differences', 'judge_reasons'].includes(msg.type);
};

export const isFlowEvent = (msg: Message): msg is FlowEvent => {
  return ['flow_event', 'error_event', 'flow_progress_update', 
          'task_processing_started', 'task_processing_complete', 'agent_announcement'].includes(msg.type);
};

export const isAgentMessage = (msg: Message): msg is AgentMessage => {
  return ['agent_input', 'agent_output', 'agent_trace', 'step_request', 
          'conductor_request', 'tool_output'].includes(msg.type);
};

export const isOSBMessage = (msg: Message): msg is OSBMessage => {
  return ['osb_query', 'osb_status', 'osb_partial', 'osb_complete', 'osb_error'].includes(msg.type);
};

export const isControlMessage = (msg: Message): msg is ControlMessage => {
  return ['run_flow', 'pull_task', 'pull_tox'].includes(msg.type);
};