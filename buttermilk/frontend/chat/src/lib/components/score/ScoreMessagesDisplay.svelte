<script lang="ts">
  import { getScoreColor } from '$lib/utils/messageUtils';
  
  // Native AgentTrace interface
  interface AgentTrace {
    timestamp: string;
    call_id: string;
    agent_id: string;
    metadata: Record<string, any>;
    outputs: any;
    agent_info: {
      agent_name: string;
      role: string;
      parameters: Record<string, any>;
    };
    session_id: string;
    inputs: {
      records: any[];
    };
  }

  export let messages: AgentTrace[];

  // Generate ASCII score bar similar to other components
  function generateScoreBar(score: number): string {
    if (score === undefined || score === null) return "░░░░░";
    
    const normalizedScore = Math.min(Math.max(score, 0), 1);
    const fullBlocks = Math.floor(normalizedScore * 5);
    const remainder = (normalizedScore * 5) - fullBlocks;
    
    let bar = "";
    
    // Add full blocks
    for (let i = 0; i < fullBlocks; i++) {
      bar += "█";
    }
    
    // Add partial block if needed  
    if (remainder > 0.25 && remainder < 0.75) {
      bar += "▌";
    } else if (remainder >= 0.75) {
      bar += "█";
    }
    
    // Add empty blocks
    const emptyBlocks = 5 - bar.length;
    for (let i = 0; i < emptyBlocks; i++) {
      bar += "░";
    }
    
    return bar;
  }

  // Format timestamp for terminal display
  function formatTimestamp(): string {
    const now = new Date();
    return now.toISOString().replace(/\.\d{3}Z$/, '');
  }

  // Process AgentTrace objects into display format
  $: processedMessages = messages.map(trace => {
    const outputs = trace.outputs || {};
    const agentName = trace.agent_info?.agent_name || trace.agent_id;
    const model = trace.agent_info?.parameters?.model || '';
    const role = trace.agent_info?.role || 'unknown';
    
    // Extract data from outputs
    const content = outputs.conclusion || outputs.content || JSON.stringify(outputs);
    const reasoning = Array.isArray(outputs.reasons) ? outputs.reasons.join(' ') : (outputs.reasons || '');
    const prediction = outputs.prediction || false;
    const confidence = outputs.confidence || 'medium';
    
    // Convert confidence to numeric score
    const confidenceMap: Record<string, number> = { high: 0.9, medium: 0.7, low: 0.5 };
    const score = typeof confidence === 'string' ? confidenceMap[confidence] || 0.7 : confidence;
    
    return {
      agent: `${agentName}-${model}`,
      type: role.toLowerCase(),
      content,
      score,
      reasoning,
      prediction,
      timestamp: trace.timestamp,
      call_id: trace.call_id
    };
  });

  // Get agent identifier with color
  function getAgentColor(agent: string): string {
    // Color coding based on agent type
    if (agent.includes('GPT')) return '#00ff00';
    if (agent.includes('Claude')) return '#00ffff';
    if (agent.includes('Gemini')) return '#ff8844';
    if (agent.includes('LLaMA')) return '#ffaa00';
    return '#fff';
  }

  // Get message type icon
  function getTypeIcon(type: string): string {
    switch(type) {
      case 'judge': return '⚖️';
      case 'synth': return '⚡';
      case 'assessment': return '📊';
      default: return '🤖';
    }
  }
</script>

<div class="score-messages-display">
  {#if processedMessages && processedMessages.length > 0}
    <div class="messages-container">
      {#each processedMessages as message, index}
        {@const scoreColor = getScoreColor(message.score)}
        {@const agentColor = getAgentColor(message.agent)}
        
        <div class="message-terminal">
          <!-- Message header in IRC style -->
          <div class="message-header">
            <span class="timestamp">[{formatTimestamp()}]</span>
            <span class="agent-name" style="color: {agentColor}">
              {getTypeIcon(message.type)} {message.agent}:
            </span>
            <span class="message-score" style="color: {scoreColor}">
              [{(message.score * 100).toFixed(0)}%] {generateScoreBar(message.score)}
            </span>
          </div>
          
          <!-- Message content -->
          <div class="message-content">
            <div class="content-body">
              {message.content}
            </div>
            
            {#if message.reasoning}
              <div class="reasoning-section">
                <div class="reasoning-header">
                  <span class="reasoning-label">REASONING:</span>
                </div>
                <div class="reasoning-content">
                  {message.reasoning}
                </div>
              </div>
            {/if}
          </div>
          
          <!-- Message footer with metadata -->
          <div class="message-footer">
            <span class="meta-item">
              TYPE: <span class="meta-value">{message.type.toUpperCase()}</span>
            </span>
            <span class="meta-item">
              CONFIDENCE: <span class="meta-value" style="color: {scoreColor}">
                {message.score >= 0.8 ? 'HIGH' : message.score >= 0.6 ? 'MEDIUM' : 'LOW'}
              </span>
            </span>
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="no-messages">
      <div class="no-messages-icon">💬</div>
      <div class="no-messages-text">No AI model responses available for this record.</div>
      <div class="no-messages-subtext">This feature will be populated when the API endpoint becomes available.</div>
    </div>
  {/if}
</div>

<style>
  .score-messages-display {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }

  .messages-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .message-terminal {
    background-color: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }

  .message-header {
    background-color: rgba(255, 255, 255, 0.02);
    padding: 0.5rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.9rem;
  }

  .timestamp {
    color: #666;
    font-size: 0.8rem;
  }

  .agent-name {
    font-weight: bold;
    min-width: 120px;
  }

  .message-score {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: bold;
  }

  .message-content {
    padding: 1rem;
  }

  .content-body {
    color: #e0e0e0;
    line-height: 1.6;
    margin-bottom: 1rem;
    background-color: rgba(255, 255, 255, 0.02);
    padding: 1rem;
    border-left: 3px solid rgba(0, 255, 255, 0.3);
  }

  .reasoning-section {
    margin-top: 1rem;
  }

  .reasoning-header {
    margin-bottom: 0.5rem;
  }

  .reasoning-label {
    color: #ffaa00;
    font-weight: bold;
    font-size: 0.85rem;
  }

  .reasoning-content {
    color: #ccc;
    font-size: 0.9rem;
    line-height: 1.5;
    background-color: rgba(255, 170, 0, 0.05);
    padding: 0.75rem;
    border-left: 3px solid rgba(255, 170, 0, 0.3);
    font-style: italic;
  }

  .message-footer {
    background-color: rgba(0, 0, 0, 0.2);
    padding: 0.5rem 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    gap: 2rem;
    font-size: 0.8rem;
  }

  .meta-item {
    color: #888;
  }

  .meta-value {
    font-weight: bold;
  }

  .no-messages {
    text-align: center;
    padding: 3rem;
    background-color: rgba(0, 0, 0, 0.3);
    border: 1px dashed rgba(255, 255, 255, 0.2);
    border-radius: 4px;
  }

  .no-messages-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
  }

  .no-messages-text {
    color: #ccc;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
  }

  .no-messages-subtext {
    color: #666;
    font-size: 0.9rem;
    font-style: italic;
  }
</style>