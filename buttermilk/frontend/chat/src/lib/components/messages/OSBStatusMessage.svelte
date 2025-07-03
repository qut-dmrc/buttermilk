<!-- 
OSB Status Message Component

Displays real-time status updates during OSB query processing.
Shows current agent being processed, progress indicators, and
processing stages in a compact terminal-style format.
-->

<script lang="ts">
  import { getAgentStyle } from '$lib/utils/messageUtils';
  import type { Message } from '$lib/utils/messageUtils';

  export let message: Message;

  // Extract status data from message content
  $: statusData = typeof message.content === 'object' ? message.content : JSON.parse(message.content || '{}');
  $: agentStyle = getAgentStyle('OSB_STATUS');

  // Progress bar styling
  $: progressPercentage = statusData.progress_percentage || 0;
  $: progressBarClass = progressPercentage < 30 ? 'bg-info' : 
                       progressPercentage < 70 ? 'bg-warning' : 'bg-success';

  // Status icon mapping
  $: statusIcon = getStatusIcon(statusData.status);

  function getStatusIcon(status: string): string {
    const icons = {
      'initializing': '🔄',
      'processing_request': '📝',
      'agent_processing': '🤖',
      'routing_to_researcher': '🔍',
      'routing_to_policy_analyst': '📋',
      'routing_to_fact_checker': '✅',
      'routing_to_explorer': '🔎',
      'executing': '⚡',
      'synthesis_ready': '📊',
      'completed': '✅',
      'error': '❌'
    };
    return icons[status] || '•';
  }

  // Format status message for display
  $: displayMessage = statusData.message || formatStatusMessage(statusData.status, statusData.agent);

  function formatStatusMessage(status: string, agent?: string): string {
    if (agent) {
      return `Processing with ${agent} agent...`;
    }
    
    const messages = {
      'initializing': 'OSB flow initialized',
      'processing_request': 'Processing OSB request',
      'agent_processing': 'Multi-agent analysis in progress',
      'executing': 'Executing OSB workflow',
      'synthesis_ready': 'Synthesizing agent responses',
      'completed': 'OSB analysis completed',
      'error': 'Error in OSB processing'
    };
    return messages[status] || status.replace(/_/g, ' ');
  }

  // Determine if this is an agent-specific status
  $: isAgentStatus = statusData.agent && statusData.status === 'agent_processing';
  $: agentColor = getAgentColor(statusData.agent);

  function getAgentColor(agent?: string): string {
    const colors = {
      'researcher': '#ffd700',      // Gold
      'policy_analyst': '#87ceeb',  // Sky blue
      'fact_checker': '#98fb98',    // Pale green
      'explorer': '#dda0dd'         // Plum
    };
    return colors[agent] || '#ffffff';
  }
</script>

<div class="status-message-wrapper mb-1">
  <div class="d-flex align-items-center">
    <!-- Status icon and timestamp -->
    <span class="status-icon me-2">{statusIcon}</span>
    <span class="text-muted small timestamp">
      {new Date(message.timestamp || Date.now()).toLocaleTimeString()}
    </span>

    <!-- Main status message -->
    <span class="status-text mx-2" class:agent-specific={isAgentStatus}>
      {#if isAgentStatus}
        <span class="agent-name" style="color: {agentColor};">
          [{statusData.agent.toUpperCase()}]
        </span>
      {/if}
      {displayMessage}
    </span>

    <!-- Progress indicator -->
    {#if statusData.progress_percentage !== undefined}
      <div class="progress-container ms-auto">
        <div class="progress progress-sm">
          <div 
            class="progress-bar {progressBarClass}" 
            role="progressbar" 
            style="width: {progressPercentage}%"
            aria-valuenow={progressPercentage} 
            aria-valuemin="0" 
            aria-valuemax="100"
          >
          </div>
        </div>
        <span class="progress-text small text-muted ms-2">
          {Math.round(progressPercentage)}%
        </span>
      </div>
    {:else if statusData.estimated_completion}
      <span class="estimated-time small text-muted ms-auto">
        ETA: {statusData.estimated_completion}
      </span>
    {/if}
  </div>

  <!-- Additional details for certain status types -->
  {#if statusData.status === 'error' && statusData.error_message}
    <div class="error-details mt-1 ms-4">
      <small class="text-danger">Error: {statusData.error_message}</small>
    </div>
  {/if}

  {#if statusData.status === 'agent_processing' && statusData.agent}
    <div class="agent-details mt-1 ms-4">
      <small class="text-muted">
        Analyzing content with {statusData.agent} capabilities...
      </small>
    </div>
  {/if}
</div>

<style>
  .status-message-wrapper {
    font-family: 'Courier New', monospace;
    background-color: rgba(0, 15, 30, 0.4);
    border-left: 3px solid rgba(100, 150, 200, 0.5);
    padding: 6px 10px;
    margin: 2px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.9em;
  }

  .status-icon {
    font-size: 1em;
    display: inline-block;
    width: 1.2em;
    text-align: center;
  }

  .timestamp {
    font-family: 'Courier New', monospace;
    opacity: 0.7;
  }

  .status-text {
    flex-grow: 1;
    color: #e0e0e0;
  }

  .status-text.agent-specific {
    font-weight: 500;
  }

  .agent-name {
    font-family: 'Courier New', monospace;
    font-weight: bold;
    letter-spacing: 0.5px;
  }

  .progress-container {
    display: flex;
    align-items: center;
    min-width: 100px;
  }

  .progress {
    width: 80px;
    height: 4px;
    background-color: rgba(255, 255, 255, 0.1);
  }

  .progress-sm {
    height: 4px;
  }

  .progress-text {
    font-family: 'Courier New', monospace;
    min-width: 30px;
  }

  .estimated-time {
    font-family: 'Courier New', monospace;
    opacity: 0.8;
  }

  .error-details,
  .agent-details {
    border-left: 2px solid rgba(255, 255, 255, 0.1);
    padding-left: 8px;
  }

  /* Animation for status changes */
  .status-message-wrapper {
    animation: statusFadeIn 0.3s ease-in-out;
  }

  @keyframes statusFadeIn {
    from {
      opacity: 0;
      transform: translateX(-10px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }

  /* Hover effect for interactive elements */
  .status-message-wrapper:hover {
    background-color: rgba(0, 20, 40, 0.6);
    border-left-color: rgba(100, 150, 200, 0.8);
  }
</style>