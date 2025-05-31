<script lang="ts">
  import { getAgentStyle, type Message } from '$lib/utils/messageUtils';

  // Props
  export let message: Message;
  export let expanded = false;
  
  // Derived values
  $: timestamp = message.timestamp ? new Date(message.timestamp).toISOString().replace(/\.\d{3}Z$/, '') : '';
  $: agentStyle = getAgentStyle(message.agent_info?.agent_name || '');
  $: messageType = message.type || 'unknown';
  $: agentName = message.agent_info?.agent_id || messageType.toUpperCase() || 'SYSTEM';
  $: error = message.outputs?.error;
</script>

<div class="container message-line">
  <!-- <span class="timestamp">[{timestamp}]</span> -->
  <div class="nick-container col-sm-1 col-md-1 col-lg-1">
    <span class="agent-nick">
      <slot name="agentNick">[{agentName}]</slot>
    </span>
    <span class="agent-metadata">
        <!-- Default metadata display -->
        {#if message.agent_info?.parameters?.model}<i class="bi bi-cpu"></i> {message.agent_info.parameters.model}{/if}
        {#if message.agent_info?.parameters?.template}<i class="bi bi-file-earmark-text"></i> {message.agent_info.parameters.template}{/if}
        <slot name="messagePrefix">
          {#if message.agent_info?.parameters?.criteria}<i class="bi bi-list-check"></i> {message.agent_info.parameters.criteria}{/if}
        </slot>
        {#if message.tracing_link}<a href={message.tracing_link} target="_blank" rel="noopener noreferrer"><i class="bi bi-link"></i>[trace]</a>{/if}
    </span>
  </div>
  <div class="message-text col-sm-10">
    <span class="message-body">
      {#if error}<span class="error-message">Error: {error}</span>{/if}
      <slot name="messageContent">
        {message.preview || JSON.stringify(message.outputs?.content || message.outputs || {})}
      </slot>
    </span>
    <div class="message-expanded">
      <slot name="messageExpanded">
        <!-- Expandable content goes here -->
      </slot>
    </div>
  </div>
</div>
