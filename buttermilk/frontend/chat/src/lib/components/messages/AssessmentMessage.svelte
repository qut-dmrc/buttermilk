<script lang="ts">
  import { slide } from 'svelte/transition';
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import { getAgentStyle, getAgentEmoji, getScoreColor, type Message } from '$lib/utils/messageUtils';
  import type { Tooltip } from 'bootstrap';
  
  // Import common expandable details styles
  import '$lib/styles/expandable-details.scss';

  // Props
  export let message: Message;
  export let expanded = false;

  // Local state
  let tooltipElement: HTMLElement | null = null;
  let tooltipInstance: Tooltip | null = null;
  
  // Lifecycle
  onMount(() => {
    if (browser && tooltipElement) {
      // Only import and use Bootstrap JS on the client
      import('bootstrap/dist/js/bootstrap.bundle.min.js').then(bootstrap => {
        // Ensure the element is available before creating the tooltip
        if (tooltipElement) {
          // Assign to the component-level variable
          tooltipInstance = new bootstrap.Tooltip(tooltipElement, { 
            trigger: 'hover' // Explicitly set trigger if needed, default is hover/focus
          });
        }
      });
    }
  });

  onDestroy(() => {
    // Dispose the tooltip instance when the component is destroyed
    if (tooltipInstance) {
      tooltipInstance.dispose();
      tooltipInstance = null; // Clear the reference
    }
  });

  // Get agent styling using agent_info if available
  $: agentStyle = getAgentStyle(message.agent_info?.agent_name || 'System');
  
  // Format timestamp - strip milliseconds part
  $: formattedTimestamp = message.timestamp ? formatTimestamp(message.timestamp) : '';
  
  function formatTimestamp(timestamp: string): string {
    try {
      const date = new Date(timestamp);
      return date.toISOString().replace(/\.\d{3}Z$/, '');
    } catch (e) {
      return timestamp;
    }
  }
  
  // Get the display name
  $: displayName = message.agent_info?.agent_name || 'System';
</script>

<!-- We don't want assessment messages to appear in the terminal, so we'll only render
  them in the sidebar. -->
{#if expanded}
  <div class="sidebar-card mt-2 mb-3">
    <div class="card">
      <div class="card-header d-flex justify-content-between align-items-center" 
           style="background-color: {agentStyle.background}; border-color: {agentStyle.border};">
        <span><small>{displayName}</small></span>
        <span 
          bind:this={tooltipElement}
          class="badge bg-light text-dark border"
          style="border-color: {getScoreColor(message.outputs.correctness)} !important;"
          data-bs-toggle="tooltip"
          data-bs-placement="left"
          title="Score: {message.outputs.score_text}"
        >
          <i class="bi bi-star-fill" style="color: {getScoreColor(message.outputs.correctness)};"></i>
        </span>
      </div>
      
      <div class="card-body p-2">
        <div class="tiny-text">
          <div>Assessed: <span class="text-muted">{message.outputs.assessed_agent_id.split('-')[0]}</span></div>
          <div>Score: <span style="color: {getScoreColor(message.outputs.correctness)};">{message.outputs.score_text}</span></div>
          
          <!-- Assessment details in sidebar -->
          {#if message.outputs.assessments && message.outputs.assessments.length > 0}
            <div class="assessment-details mt-1">
              <div class="assessment-header">Assessment Details:</div>
              <ul class="assessment-list">
                {#each message.outputs.assessments as assessment}
                  <li>{assessment}</li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .sidebar-card .card {
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .sidebar-card .card-header {
    padding: 0.5rem;
    font-size: 0.85rem;
  }
  
  .tiny-text {
    font-size: 0.7rem;
    line-height: 1.2;
  }
  
  /* Assessment details in sidebar */
  .assessment-header {
    font-weight: bold;
    margin-top: 5px;
    margin-bottom: 3px;
    font-size: 0.75rem;
  }
  
  .assessment-list {
    list-style-type: none;
    padding-left: 0;
    margin-bottom: 0;
    font-size: 0.65rem;
  }
  
  .assessment-list li {
    margin-bottom: 2px;
    padding-left: 5px;
    border-left: 2px solid #5fad4e;
  }
  
  /* Bootstrap Icons Adjustments */
  .bi {
    vertical-align: -0.125em;
  }
</style>
