// Agent type styling
export const AGENT_STYLES = {
  default: { color: "#6c757d", background: "#f8f9fa", border: "#dee2e6" },
  judge: { color: "#495057", background: "#e9ecef", border: "#ced4da" },
  scorer: { color: "#212529", background: "#f8f9fa", border: "#adb5bd" },
  assistant: { color: "#0d6efd", background: "#e7f1ff", border: "#b6d4fe" },
  describer: { color: "#6610f2", background: "#eee6ff", border: "#d0bfff" },
  fetch: { color: "#fd7e14", background: "#fff3cd", border: "#ffecb5" },
  imagegen: { color: "#d63384", background: "#f7d6e6", border: "#efadce" },
  reasoning: { color: "#20c997", background: "#d1f2eb", border: "#a3e4d7" },
  scraper: { color: "#6f42c1", background: "#e6d9f2", border: "#d2b5e8" },
  spy: { color: "#212529", background: "#e2e3e5", border: "#c9cccf" },
  synthesiser: { color: "#0c4128", background: "#d0f0e0", border: "#a0d0b0" },
  tool: { color: "#198754", background: "#d1e7dd", border: "#badbcc" },
  instructions: { color: "#0dcaf0", background: "#cff4fc", border: "#9eeaf9" },
  record: { color: "#6c757d", background: "#f8f9fa", border: "#dee2e6" },
  summary: { color: "#007bff", background: "#e7f1ff", border: "#b6d4fe" },
  researcher: { color: "#5fadaa", background: "#d1f0f0", border: "#a0d0d0" },
};


export function getModelColor(modelName: string | undefined): string {
  if (!modelName) return '#aaaaaa';
  
  const modelLower = modelName.toLowerCase();
  if (modelLower.includes('gpt-4')) return '#10a37f';
  if (modelLower.includes('gpt-3.5')) return '#1f85de';
  if (modelLower.includes('claude')) return '#8e44ad';
  if (modelLower.includes('gemini')) return '#f39c12';
  if (modelLower.includes('llama')) return '#e74c3c';
  if (modelLower.includes('falcon')) return '#3498db';
  
  let hash = 0;
  if (modelName) {
    for (let i = 0; i < modelName.length; i++) {
      hash = modelName.charCodeAt(i) + ((hash << 5) - hash);
    }
  }
  
  const c = (hash & 0x00FFFFFF)
    .toString(16)
    .toUpperCase()
    .padStart(6, '0');
  
  return `#${c}`;
}

export function getModelIdentifier(message: Message): string {
  if (!message.agent_info?.parameters?.model) return '';

  const idLower = message.agent_info.parameters.model.toLowerCase();
  if (idLower.includes('gpt4')) return 'GPT4';
  if (idLower.includes('gpt3')) return 'GPT3';
  if (idLower.includes('sonnet')) return 'CLDE';
  if (idLower.includes('gemini')) return 'GEMN';
  if (idLower.includes('llama')) return 'LLMA';
  return 'UNKN';
}

export function getRoleIdentifier(message: Message): string {
  try {
    const firstWord = message.agent_info?.parameters?.role.substring(0, 4).toUpperCase();
    return `│${firstWord}│ `;
  } catch (e) {
    return 'SYST'
  }
}

// Get agent style based on name
export function getAgentStyle(agentName: string) {
  const lowerName = agentName.toLowerCase();
  for (const [type, style] of Object.entries(AGENT_STYLES)) {
    if (lowerName.includes(type)) {
      return style;
    }
  }
  return AGENT_STYLES.default;
}
// Get agent emoji based on name
export function getAgentEmoji(agentName: string) {
  return '🤖'; // Default to robot emoji if not found
}

// Format uncertainty indicator
export function formatUncertainty(uncertainty: string) {
  let level = 0;
  let color = "#dc3545"; // Default red for high uncertainty

  if (typeof uncertainty === 'string') {
    if (uncertainty.toLowerCase().includes('high')) level = 3;
    else if (uncertainty.toLowerCase().includes('medium')) level = 2;
    else if (uncertainty.toLowerCase().includes('low')) level = 1;
  }

  if (level === 1) color = "#28a745"; // Green for low uncertainty
  else if (level === 2) color = "#ffc107"; // Yellow for medium uncertainty

  return {
    icon: level === 1 ? "bi-hand-thumbs-up" : level === 2 ? "bi-question-circle" : "bi-exclamation-triangle",
    color: color,
    text: uncertainty,
    level: level
  };
}

// Score Badge color
export function getScoreColor(score: string | number | null | undefined) {
  if (score === null || score === undefined) return "#6c757d"; // Default gray

  const numScore = typeof score === 'string' ? parseFloat(score) : score;

  if (numScore > 0.8) return "#28a745"; // Strong green
  if (numScore > 0.6) return "#5cb85c"; // Light green
  if (numScore > 0.4) return "#ffc107"; // Yellow
  if (numScore > 0.2) return "#ff9800"; // Orange
  return "#dc3545"; // Red
}

// Helper function to format time as HH:MM:SS
function formatTime(date: Date): string {
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
}

// --- Message Type Enum ---
export type MessageType =
  | 'chat_message'
  | 'record'
  | 'ui_message'
  | 'user'
  | 'manager_response'
  | 'system_message'
  | 'qual_results'
  | 'differences'
  | 'judge_reasons'
  | 'system_error'
  | 'system_update'
  | 'assessments'  
  | 'judge' 
  | 'research_result'
  | 'summary_result';

// Agent info record structure
export interface AgentParams {
  model?: string;        // Model used by agent (e.g., gpt-4)
  template?: string;     // Template name used by agent
  criteria?: string;     // Criteria used by agent for evaluation
  [key: string]: any; // Allow for additional parameters
}
export interface AgentInfo {
  agent_id: string;
  session_id: string;
  role: string;
  agent_name: string;
  description: string;
  parameters?: AgentParams; // Parameters used by agent
}

// --- Type definitions ---

// Basic Message interface
export interface Message {
  preview?: string;
  timestamp: string;
  outputs?: any; // Keep outputs as any for flexibility, access properties based on 'type'
  agent_info?: AgentInfo;
  type: MessageType;
  message_id: string;
  tracing_link?: string;
}

// Judge message types
export interface JudgeReasons {
  prediction: boolean; // violates or not
  conclusion: string;
  reasons: string[];
  uncertainty?: string;
}

// Assessment message types
export interface Assessments {
  assessed_agent_id: string;
  assessed_call_id: string;
  correctness: string;
  score_text: string;
  assessments: string[];
}

// Record message types
export interface RecordMetadata {
  title?: string;
  source?: string;
  date?: string;
  author?: string;
  url?: string;
  outlet?: string;
  [key: string]: any; // Allow for additional metadata fields
}

export interface RecordData {
  content: string; // Markdown formatted content
  metadata: RecordMetadata;
  text?: string; // Can be ignored as per requirements
}

// Researcher data types
export interface LiteratureItem {
  summary: string;
  citation: string;
}

export interface ResearcherData {
  literature: LiteratureItem[];
  response: string;
}

// Differences data types
export interface Expert {
  name: string;
  answer_id: string;
}

export interface Position {
  experts: Expert[];
  position: string;
}

export interface Divergence {
  topic: string;
  positions: Position[];
}

export interface DifferencesData {
  conclusion: string;
  divergences: Divergence[];
}

// Summary results type for aggregating predictions and scores
export interface PredictionSummary {
  agent_id: string;
  agent_name?: string;
  message_id: string;
  prediction: boolean;
  uncertainty?: string;
  score?: number;
  assessments?: {
    agent_id: string;
    agent_name?: string;
    score: number;
    text?: string;
  }[];
}

export interface SummaryResult {
  type: 'summary_result',
  run_id: string;
  record_id?: string;
  flow_id?: string;
  timestamp?: string;
  predictions: PredictionSummary[];
  avg_score?: number;
  agreement_rate?: number;
}

export interface TaskProcessingStarted {
  type: "TaskProcessingStarted",
  role: "MANAGER",
  agent_id: "web socket",
}
export interface TaskProcessingFinished {
  type: "TaskProcessingFinished",
  role: "MANAGER",
  agent_id: "web socket"
}

// Manager Request types (from Python definition)

export interface UIMessage {
  content: string;
  options?: boolean | string[] | null; // Options for the manager to choose from
  confirm?: boolean|null;
  selection?: string | null;
  toughts?: string;
}

// Manager Response types (from Python definition)
export interface ManagerResponse {
  type: 'manager_response',
  confirm?: boolean|null;
  halt: boolean | null;
  interrupt: boolean | null;
  content: string | null;
  selection: string | null;
  human_in_loop?: boolean | null;
  params?: Record<string, any> | null; // Using Record for Mapping[str, Any]
}


export interface SystemUpdate {
  source: string;
  step_name: string;
  status: string;
  message?: string;
  timestamp: string;
  waiting_on?: Record<string, number>;
}
export interface SystemMessage {
  type: 'system_message' | 'system_error',
  content: string;
  timestamp?: string;
}

// Union type for all message types
// Adding a generic message type for fallback cases
export interface GenericMessage {
  type: string;
  content?: string;
  timestamp?: string;
  [key: string]: any; // Allow for additional properties
}

export type WebSocketData = SystemUpdate  | SystemMessage | UIMessage | RecordData | ManagerResponse | SummaryResult | GenericMessage;

export function isSystemUpdate(data: any): data is SystemUpdate {
  return (
    typeof data === 'object' &&
    data !== null &&
    data.type === 'system_update'
  );
}


// Type guard for Summary Result
export function isSummaryResult(data: any): data is SummaryResult {
  return (
    typeof data === 'object' &&
    data !== null &&
    data.type === 'summary_result' &&
    typeof data.run_id === 'string' && // Check for mandatory run_id
    Array.isArray(data.predictions)    // Keep check for mandatory predictions
  );
}

// Type guard for Record
export function isRecord(data: any): data is RecordData {
  return (
    typeof data === 'object' &&
    data !== null &&
    data.type === 'record'
  );
}

// Type guard for Researcher data
export function isResearcher(data: any): data is ResearcherData {
  return (
    typeof data === 'object' &&
    data !== null &&
    data.type === 'researcher'
  );
}

// Type guard for JudgeReasons
export function isJudgeReasons(data: any): data is JudgeReasons {
  return (
    typeof data === 'object' &&
    data !== null &&
    typeof data.prediction === 'boolean' &&
    typeof data.conclusion === 'string' &&
    Array.isArray(data.reasons)
  );
}

// Type guard for Assessment
export function isAssessment(data: any): data is Assessments {
  return (
    typeof data === 'object' &&
    data !== null &&
    data.type === 'assessment' &&
    typeof data.correctness === 'string' &&
    Array.isArray(data.assessments)
  );
}

// Type guard for Differences
export function isDifferences(data: any): data is DifferencesData {
  return (
    typeof data === 'object' &&
    data !== null &&
    typeof data.conclusion === 'string' &&
    Array.isArray(data.divergences) &&
    data.divergences.every(
      (div: any) =>
        typeof div.topic === 'string' &&
        Array.isArray(div.positions) &&
        div.positions.every(
          (pos: any) =>
            typeof pos.position === 'string' &&
            Array.isArray(pos.experts)
        )
    )
  );
}


/**
 * Normalizes raw WebSocket messages to a consistent structure
 */
export function normalizeWebSocketMessage(data: any): Message {
  try {
    // Validate basic message structure
    if (typeof data !== 'object' || data === null) {
      console.error('Invalid message format, not an object:', data);
      return createErrorMessage(`Invalid message format: ${JSON.stringify(data)}`);
    }

    // Create normalized message structure
    const normalizedMessage: Message = {
      type: data.type as MessageType,
      message_id: data.message_id,
      preview: data.preview ||'',
      timestamp: data.timestamp,
      outputs: data.outputs || null,
      agent_info: data.agent_info as AgentInfo|| null,
    };

    return normalizedMessage;
  } catch (e) {
    console.error('Error normalizing WebSocket message:', e);
    return createErrorMessage(`Error processing message: ${e}`);
  }
}

/**
 * Helper function to create error messages with consistent format
 */
function createErrorMessage(content: string): Message {
  return {
    type: 'system_error',
    message_id: 'SYSTEM_ERROR' + Math.random().toString(36).substring(2, 15),
    preview: content,
    timestamp: new Date().toISOString()
  };
}
