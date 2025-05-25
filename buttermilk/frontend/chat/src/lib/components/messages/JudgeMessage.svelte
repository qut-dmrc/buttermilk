<script lang="ts">
  import { slide } from 'svelte/transition';
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import { formatUncertainty, getModelColor, getModelIdentifier, getRoleIdentifier, getScoreColor, type Message } from '$lib/utils/messageUtils';
  import { calculateAverageScore } from '$lib/utils/scoreUtils';
  import { messageStore } from '$lib/stores/messageStore';
  import type { Tooltip } from 'bootstrap';
  
  import '$lib/styles/expandable-details.scss';
	import BasicMessage from './BasicMessage.svelte';
  

  export let message: Message;
  export let expanded = false;

  let showDetails = false;
  let showAssessments = false;
  let tooltipElement: HTMLElement;
  let tooltipInstance: Tooltip | null = null;
  let assessments: Message[] = [];
  let averageScore: number | null = null;
  // Reactive declaration for assessments
  $: {
    // Log when the messageStore updates and the reactive block runs
    console.debug(`[JudgeMessage ${message.message_id}] Reactive block triggered by $messageStore update.`);
    // Log the current state of the messageStore (be cautious with large stores)
    // console.log(`[JudgeMessage ${message.message_id}] Current $messageStore content:`, $messageStore);

    // Filter the message store for assessment messages related to this judge message
    assessments = $messageStore.filter(m =>
      m.type === 'assessments' &&
      m.outputs?.assessed_call_id === message.message_id
    );

    // Log the result of the filtering
    console.debug(`[JudgeMessage ${message.message_id}] Filtered assessments (${assessments.length} found):`, assessments);
  }

  // Reactive declaration for averageScore
  $: {
    // Calculate the average score from the filtered assessments
    averageScore = calculateAverageScore(assessments);
    // Log the calculated average score
    console.debug(`[JudgeMessage ${message.message_id}] Calculated average score:`, averageScore);
  } 
  function scoreToBraille(score: number | undefined | null): string {
    if (score === undefined || score === null) return "⠀";
    
    const normalizedScore = Math.min(Math.max(score, 0), 1);
    
    if (normalizedScore < 0.125) return "⠠";
    if (normalizedScore < 0.25) return "⠴";
    if (normalizedScore < 0.375) return "⠶";
    if (normalizedScore < 0.5) return "⠾";
    if (normalizedScore < 0.625) return "⡾";
    if (normalizedScore < 0.75) return "⣾";
    if (normalizedScore < 0.875) return "⣿";
    return "⣿";
  }
  
  function generateScoreBar(score: number | undefined | null): string {
    if (score === undefined || score === null) return "⠀⠀⠀⠀⠀";
    
    return scoreToBraille(score).repeat(5);
  }
  
  function generateMultiScoreBar(assessmentMessages: Message[]): string {
    if (!assessmentMessages || assessmentMessages.length === 0) {
      return "⠀⠀⠀⠀⠀";
    }
    
    let bar = "";
    
    assessmentMessages.forEach(msg => {
      if (msg.outputs && msg.outputs.correctness !== undefined) {
        const score = parseFloat(msg.outputs.correctness);
        if (!isNaN(score)) {
          bar += scoreToBraille(score);
        } else {
          bar += "⠀";
        }
      } else {
        bar += "⠀";
      }
    });
    
    return bar;
  }
  
  onMount(() => {
    if (browser && tooltipElement) { // Check if in browser and element is bound
      import('bootstrap/dist/js/bootstrap.bundle.min.js').then(bootstrap => {
        console.log('Bootstrap JS loaded on client');
        tooltipInstance = new bootstrap.Tooltip(tooltipElement, { 
          trigger: 'hover'
        });
      });
    }
  });

  onDestroy(() => {
    if (tooltipInstance) {
      tooltipInstance.dispose();
      tooltipInstance = null;
    }
  });
  
  $: modelName = getModelIdentifier(message) || '';
  $: modelBasedColor = getModelColor(message.agent_info?.parameters?.model);
  
  $: prediction = message.outputs?.prediction;
  $: uncertainty = message.outputs?.uncertainty ? formatUncertainty(message.outputs.uncertainty) : undefined;
  
  function formatTimestamp(timestamp: string): string {
    try {
      const date = new Date(timestamp);
      return date.toISOString().replace(/\.\d{3}Z$/, '');
    } catch (e) {
      return timestamp;
    }
  }
  
  function toggleDetails() {
    showDetails = !showDetails;
  }
  
  function toggleAssessments() {
    showAssessments = !showAssessments;
  }
  
  $: displayName = message.agent_info?.agent_name || 'UNKWN';
</script>

<div class="message-terminal" style="color: {modelBasedColor}">
  <BasicMessage message={message}>
    <svelte:fragment slot="messagePrefix">
        <i class="bi bi-cpu"></i>|{modelName}|
        <i class="bi bi-file-earmark-text"></i>{message.agent_info?.parameters?.template}
        <i class="bi bi-list-check"></i>{message.agent_info?.parameters?.criteria}
        {#if prediction !== undefined}<i class="bi {prediction ? 'bi-x-circle' : 'bi-check-circle'}"></i>{/if}
        {#if uncertainty}<i class="bi {uncertainty.icon}">{uncertainty.text}</i>{/if}
        <br>
        <span class="avg-score" style="color: {getScoreColor(averageScore)}">
          Average: {averageScore !== null ? (averageScore * 100).toFixed(0) + '%' : 'N/A'} 
          {generateMultiScoreBar(assessments)}
        </span>
    </svelte:fragment>

    <svelte:fragment slot="messageContent">
      <div class="content-inline">
        {message.outputs?.conclusion}
        <!-- Content Toggle (Inline button) -->
        <button
          class="content-toggle-inline"
          on:click={toggleDetails}
          title={showDetails ? "Hide reasons" : "Show reasons"}
        >  {showDetails ? '[-]' : '[+]'} reasons
        </button>   
        <button
          class="content-toggle-inline"
          on:click={toggleAssessments}
          title="assessments"
        >  
          {showAssessments ? '[-]' : '[+]'} assessments ({assessments.length})
        </button>   
      </div>
    </svelte:fragment>

    <svelte:fragment slot="messageExpanded">
      {#if showDetails}
        <div class="judge-message">
          <div class="expanded-content markdown-content" transition:slide>
                <i class="bi bi-chat-square-quote"></i>
                <ol>
                  {#each message.outputs.reasons as reason}
                    <li>{reason}</li>
                  {/each}
                </ol>
          </div>
        </div>

      {/if}
      {#if showAssessments}
        <div class="assessment-list assessment-content tiny-text" transition:slide>

          {#each assessments as assessment}
            <div class="assessment-item">
              <div class="assessment-agent">
                {assessment.agent_info?.agent_name} 
                <span class="assessment-score" style="color: {getScoreColor(assessment.outputs?.correctness)}">
                  {assessment.outputs?.correctness ? (parseFloat(assessment.outputs.correctness) * 100).toFixed(0) + '%' : 'N/A'} {scoreToBraille(assessment.outputs?.correctness ? parseFloat(assessment.outputs.correctness) : undefined)}
                </span>
              </div>
              <ul class="assessment-reasons">
                {#each assessment.outputs?.assessments || [] as reason}
                  <li style="color: {reason.correct ? '#e0fbfc' : '#dc3545'}">{reason.feedback}</li>
                {/each}
              </ul>
            </div>
          {/each}
        </div>
      {/if}
    </svelte:fragment>
  </BasicMessage>
</div>

<!-- Sidebar Card (only if this message should appear in sidebar) -->
{#if expanded}
  <div class="sidebar-card mt-2 mb-3">
    <div class="card">
      <div class="card-header d-flex justify-content-between align-items-center" 
           style="background-color: rgba(0, 0, 0, 0.2); border-color: {modelBasedColor}; border-left: 3px solid {modelBasedColor};">
        <span>
          <span class="agent-tag">{agentIdentifier}</span> <small>{displayName}</small>
        </span>
        <span style="color: {prediction ? '#dc3545' : '#28a745'}; font-weight: bold;">
          {prediction ? 'VIOLATION' : 'CLEAN'}
        </span>
      </div>
      
      <div class="card-body p-2">
        <div class="tiny-text">
          {#if message.agent_info?.parameters?.model}
            <div>Model: <span style="color: {modelBasedColor};">{message.agent_info?.parameters?.model}</span></div>
          {:else}
            <div>Agent ID: <span class="text-muted">{message.agent_info?.agent_id || 'Unknown'}</span></div>
          {/if}
          
          <div>Uncertainty: <span style="color: {uncertainty?.color};">{uncertainty?.text || 'N/A'}</span></div>
          
          <!-- Add criteria if available -->
          {#if message.agent_info?.parameters?.criteria}
            <div>Criteria: <span class="text-muted">{message.agent_info.parameters.criteria}</span></div>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  /* Bootstrap Icons Adjustments */
  .bi {
    vertical-align: -0.125em;
  }
</style>
