<script lang="ts">
  import { fly, slide } from 'svelte/transition';
  import { 
    type Message, 
    type SummaryResult,
    getAgentEmoji,
    getScoreColor,
    isSummaryResult
  } from '$lib/utils/messageUtils';
  
  import { flowRunning } from '$lib/stores/apiStore';
  
  // State for expandable panels
  type ExpandedPanels = {
    [key: string]: {
      details: boolean;
      reasons: boolean;
      assessments: boolean;
    }
  };
  let expandedPanels: ExpandedPanels = {};
  
  // Helper to get panel state safely with fallbacks
  function getPanelState(messageId: string | undefined, panel: 'details' | 'reasons' | 'assessments'): boolean {
    if (!messageId) return false;
    if (!expandedPanels[messageId]) return false;
    return expandedPanels[messageId][panel] || false;
  }
  
  // Helper to toggle panel state safely
  function togglePanelState(messageId: string | undefined, panel: 'details' | 'reasons' | 'assessments'): void {
    if (!messageId) return;
    
    if (!expandedPanels[messageId]) {
      expandedPanels[messageId] = { details: false, reasons: false, assessments: false };
    }
    
    expandedPanels[messageId][panel] = !expandedPanels[messageId][panel];
    expandedPanels = expandedPanels; // Trigger reactivity
  }
  
  // Props 
  export let messages: Message[] = [];
  // Theme is managed globally via document.body.className
  
  // Get summary messages for the sidebar
  $: summaryMessages = messages.filter(m => 
    m.type === 'summary_result' || 
    (m.outputs && isSummaryResult(m.outputs))
  );
  
  // Get only Judge messages for the sidebar when no summary is available
  $: judgeMessages = messages.filter(m => 
    m.outputs && m.outputs.prediction !== undefined && m.outputs.conclusion !== undefined
  );
  
  // Generate ASCII score bars
  function generateScoreBar(score: number | undefined): string {
    if (score === undefined) return "░░░░░";
    
    const normalizedScore = Math.min(Math.max(score, 0), 1);
    const fullBlocks = Math.floor(normalizedScore * 5);
    const remainder = (normalizedScore * 5) - fullBlocks;
    
    let bar = "";
    
    // Add full blocks
    for (let i = 0; i < fullBlocks; i++) {
      bar += "█";
    }
    
    // Add partial block if needed
    if (remainder > 0.1 && remainder < 0.9) {
      bar += "▌";
    } else if (remainder >= 0.9) {
      bar += "█";
    }
    
    // Add empty blocks
    const emptyBlocks = 5 - bar.length;
    for (let i = 0; i < emptyBlocks; i++) {
      bar += "░";
    }
    
    return bar;
  }

  // Get associated score messages for a given judge message ID
  function getScoresForJudgeMessage(judgeMessageId: string): Message[] {
    if (!judgeMessageId) return [];
    
    // Find messages that are assessments targeting the judge message ID
    return messages.filter(m => 
      m.outputs && 
      m.outputs.correctness !== undefined && // Ensure it's an assessment message
      m.outputs.assessed_call_id === judgeMessageId // Match the target message ID
    ); 
  }


  // --- Tooltip Action ---
  let tooltipElement: HTMLDivElement | null = null;

  function detailedScoresTooltip(node: HTMLElement, scores: Message[]) {
    let isHovering = false;

    function createTooltipElement() {
      if (!tooltipElement) {
        tooltipElement = document.createElement('div');
        tooltipElement.className = 'score-tooltip';
        document.body.appendChild(tooltipElement);
      }
    }

    function updateTooltipContent() {
      if (!tooltipElement || !scores || scores.length === 0) return;
      
      let content = '<ul class="score-list">';
      scores.forEach(score => {
        const scoreValue = score.outputs?.correctness ?? 'N/A';
        const scoreText = score.outputs?.score_text ?? scoreValue; // Use score_text if available
        const scoreColor = getScoreColor(scoreValue);
        const agentName = score.agent_info?.agent_name || 'Unknown Scorer';
        
        content += `<li>
                      <span class="scorer-name">${agentName}:</span> 
                      <span class="scorer-score" style="color: ${scoreColor};">${scoreText}</span>
                    </li>`;
      });
      content += '</ul>';
      tooltipElement.innerHTML = content;
    }

    function positionTooltip(event: MouseEvent) {
      if (!tooltipElement) return;
      // Position slightly below and substantially to the left of the cursor
      tooltipElement.style.left = `${event.pageX - 160}px`;
      tooltipElement.style.top = `${event.pageY + 10}px`;
      tooltipElement.style.display = 'block';
    }

    function handleMouseEnter(event: MouseEvent) {
      isHovering = true;
      createTooltipElement();
      updateTooltipContent();
      positionTooltip(event);
    }

    function handleMouseLeave() {
      isHovering = false;
      if (tooltipElement) {
        tooltipElement.style.display = 'none';
      }
    }
    
    node.addEventListener('mouseenter', handleMouseEnter);
    node.addEventListener('mouseleave', handleMouseLeave);
    return {
      // Update function if scores change reactively (might not be needed if scores array is stable per entry)
      update(newScores: Message[]) {
        scores = newScores;
        // If hovering when scores update, refresh the tooltip content
        if (isHovering) {
           updateTooltipContent();
        }
      },
      destroy() {
        node.removeEventListener('mouseenter', handleMouseEnter);
        node.removeEventListener('mouseleave', handleMouseLeave);
      }
    };
  }

  // Helper function to safely format percentages
  function formatPercent(value: number | null | undefined): string {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(0)}%`;
  }

  // Get the substring of an ID safely
  function safeSubstring(str: string | undefined | null, start: number, end?: number): string {
    if (!str) return 'Unknown';
    return str.substring(start, end);
  }
  
  // Clean up tooltip on component destroy
  import { onDestroy } from 'svelte';
  onDestroy(() => {
    if (tooltipElement) {
      tooltipElement.remove();
      tooltipElement = null;
    }
  });
</script>

{#if $flowRunning}
<div class="message-sidebar">
  <div class="sidebar-header">
    <h6 class="m-0">Run Summary</h6>
  </div>
  
  <div class="sidebar-content">
    <!-- If we have summary messages, show them -->
    {#if summaryMessages.length > 0}
      {#each summaryMessages as message, i (message.message_id ? `${message.message_id}_${i}` : `summary_${i}_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`)}
        {@const summaryData = message.outputs}
        <div 
          class="summary-entry" 
          in:fly={{ y: 20, duration: 300, delay: i * 50 }}
        >
          <!-- Summary Title -->
          <div class="summary-title">
            <span class="title-badge">📑</span>
            Run Summary 
            {#if summaryData.run_id}
              <span class="run-id">#{safeSubstring(summaryData.run_id, 0, 6)}</span>
            {/if}
          </div>
          
          <!-- Stats Overview -->
          <div class="summary-stats">
            <!-- Agreement Rate -->
            <div class="stat-row">
              <span class="stat-label">Agreement:</span>
              <span class="stat-value">
                {summaryData.agreement_rate ? 
                  formatPercent(summaryData.agreement_rate) : 
                  summaryData.predictions && summaryData.predictions.length > 1 ?
                    `${Math.round((Math.max(
                      summaryData.predictions.filter((p: any) => p.prediction).length,
                      summaryData.predictions.filter((p: any) => !p.prediction).length
                    ) / summaryData.predictions.length) * 100)}%` :
                    'N/A'
                }
              </span>
            </div>
            
            <!-- Average Score -->
            <div class="stat-row">
              <span class="stat-label">Avg Score:</span>
              <span class="stat-value" style="color: {getScoreColor(summaryData.avg_score)}">
                {summaryData.avg_score ? 
                  formatPercent(summaryData.avg_score) : 
                  'N/A'
                }
                <span class="ascii-bar">{generateScoreBar(summaryData.avg_score)}</span>
              </span>
            </div>
          </div>
          
          <!-- Predictions List -->
          <div class="predictions-list">
            {#if summaryData.predictions && summaryData.predictions.length > 0}
              {#each summaryData.predictions as prediction}
                {@const predictionIcon = prediction.prediction ? "✅" : "❌"}
                {@const scoreColor = getScoreColor(prediction.score)}
                
                <div class="prediction-item">
                  <span class="pred-icon">{predictionIcon}</span>
                  <span class="pred-agent">
                    {prediction.agent_info?.name || safeSubstring(prediction.agent_info?.agent_id, 0, 8)}
                  </span>
                  <span class="pred-score" style="color: {scoreColor}">
                    {prediction.score ? formatPercent(prediction.score) : "--"}
                    <span class="ascii-bar">{generateScoreBar(prediction.score)}</span>
                  </span>
                </div>
              {/each}
            {:else}
              <div class="no-predictions">No prediction data available</div>
            {/if}
          </div>
        </div>
      {/each}
    <!-- If no summaries but we have judge messages, show the individual judge entries -->
    {:else if judgeMessages.length > 0}
      {#each judgeMessages as message, i (message.message_id ? `${message.message_id}_${i}` : `judge_${i}_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`)}
        {@const scores = getScoresForJudgeMessage(message.message_id || '')}
        {@const averageScore = calculateAverageScore(scores)}
        
        <div 
          class="judge-entry" 
          in:fly={{ y: 20, duration: 300, delay: i * 50 }}
        >
          <div class="judge-info">
            <div class="judge-header">
              <div class="judge-name">
                {getAgentEmoji(message.agent_info.agent_id)}
              </div>
              <!-- Compact Score Display with Tooltip Action -->
              <!-- Terminal-style score indicator with just the ASCII bar -->
              <div class="terminal-score">
                {#if averageScore !== null && averageScore !== undefined}
                  <span 
                    class="score-indicator" 
                    style="color: {getScoreColor(averageScore)};"
                    title="Avg Score: {averageScore.toFixed(2)}"
                  >
                    {generateScoreBar(averageScore)}
                  </span>
                {:else}
                  <span class="no-scores">░░░░░</span>
                {/if}
              </div>
            </div>

            <!-- Agent Metadata Badges -->
            {#if message.agent_info}
              <div class="agent-metadata">
                {#if message.agent_info.parameters?.model}
                  <span class="metadata-badge model-badge" title="AI Model">
                    <i class="bi bi-cpu"></i> {message.agent_info.parameters?.model}
                  </span>
                {/if}
                {#if message.agent_info.parameters?.template}
                  <span class="metadata-badge template-badge" title="Template">
                    <i class="bi bi-file-text"></i> {message.agent_info.parameters?.template}
                  </span>
                {/if}

                {#if message.agent_info.parameters?.criteria}
                  <span class="metadata-badge criteria-badge" title="Evaluation Criteria">
                    <i class="bi bi-check2-square"></i> {message.agent_info.parameters?.criteria}
                  </span>
                {/if}
              </div>
            {/if}
          </div>
          
          <!-- Judge conclusion with toggle option for reasons -->
          <div class="judge-content">
            <div class="judge-decision">
              <!-- Decision indicator -->
              <span class="decision-badge {message.outputs.prediction ? 'violation' : 'clean'}">
                {message.outputs.prediction ? '✖' : '✓'}
              </span>
              
              <!-- View details toggle button -->
              <button 
                class="toggle-btn details-toggle"
                on:click={() => togglePanelState(message.message_id, 'details')}
                title="View conclusion text"
              >
                {getPanelState(message.message_id, 'details') ? '[-]' : '[+]'} Details
              </button>
            </div>
            
            <!-- Expandable conclusion text -->
            {#if getPanelState(message.message_id, 'details')}
              <div class="judge-conclusion" transition:slide={{ duration: 150 }}>
                {message.outputs.conclusion}
              </div>
            {/if}
            
            <!-- Reasons & Assessments Actions Row -->
            <div class="action-buttons">
              {#if message.outputs.reasons && message.outputs.reasons.length > 0}
                <button 
                  class="action-btn reasons-btn"
                  on:click={() => togglePanelState(message.message_id, 'reasons')}
                  class:active={getPanelState(message.message_id, 'reasons')}
                  title="View reasoning"
                >
                  <i class="bi bi-list-task"></i> Reasons
                </button>
              {/if}
              
              {#if scores && scores.length > 0}
                <button 
                  class="action-btn scores-btn"
                  on:click={() => togglePanelState(message.message_id, 'assessments')}
                  class:active={getPanelState(message.message_id, 'assessments')}
                  title="View assessment scores"
                >
                  <i class="bi bi-star"></i> Scores ({scores.length})
                </button>
              {/if}
            </div>
            
            <!-- Expandable Judge Reasons -->
            {#if getPanelState(message.message_id, 'reasons') && message.outputs.reasons && message.outputs.reasons.length > 0}
              <div class="reasons-panel" transition:slide={{ duration: 200 }}>
                <ul class="reasons-list">
                  {#each message.outputs.reasons as reason}
                    <li>{reason}</li>
                  {/each}
                </ul>
              </div>
            {/if}
            
            <!-- Assessments Section (if there are any scores) -->
            {#if scores && scores.length > 0 && getPanelState(message.message_id, 'assessments')}
              <div class="assessment-section" transition:slide={{ duration: 200 }}>
                
                <!-- Expandable Assessment Details -->
                {#if getPanelState(message.message_id, 'assessments')}
                  <div class="score-chips" transition:slide={{ duration: 150 }}>
                    {#each scores as score}
                      {@const scoreValue = score.outputs?.correctness ? parseFloat(score.outputs.correctness) : null}
                      {@const scoreColor = getScoreColor(scoreValue)}
                      {@const hasReasons = score.outputs?.assessments && score.outputs.assessments.length > 0}
                      
                      {#each [score] as scoreItem}
                        {@const modelName = score.agent_info?.parameters?.model || ''}
                        {@const identifier = getModelIdentifier(score.agent_info?.agent_name || modelName || score.agent_info?.agent_id || 'Score')}
                        {score.agent_info?.agent_name}
                        <div class="score-chip" on:click={() => {
                            // Set a unique detail key for this score
                            const detailKey = `${message.message_id}_${score.message_id}`;
                            expandedPanels[detailKey] = expandedPanels[detailKey] ? 
                              { ...expandedPanels[detailKey], details: !expandedPanels[detailKey].details } : 
                              { details: true, reasons: false, assessments: false };
                            expandedPanels = expandedPanels;
                          }}>
                          <span class="terminal-score-chip">
                            <span class="agent-tag" style="color: {scoreColor};">{identifier}</span>
                            <span class="score-indicator" style="color: {scoreColor};">
                              {generateScoreBar(scoreValue !== null ? scoreValue : undefined)}
                            </span>
                            {#if hasReasons}
                              <i class="bi bi-chevron-{expandedPanels[`${message.message_id}_${score.message_id}`]?.details ? 'up' : 'down'} expand-indicator"></i>
                            {/if}
                          </span>
                        </div>
                      {/each}
                      
                      <!-- Expandable details for this score -->
                      {#if hasReasons && expandedPanels[`${message.message_id}_${score.message_id}`]?.details}
                        <div class="score-details" transition:slide={{ duration: 150 }}>
                          <ul class="assessment-reasons">
                            {#each score.outputs.assessments as assessment}
                              <!-- Format assessment properly if it's an object -->
                              <li>
                                {#if typeof assessment === 'object'}
                                  {assessment.text || assessment.reason || (assessment.correctness !== undefined ? `Score: ${assessment.correctness}` : JSON.stringify(assessment))}
                                {:else}
                                  {assessment}
                                {/if}
                              </li>
                            {/each}
                          </ul>
                        </div>
                      {/if}
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </div>
      {/each}
    <!-- If neither summaries nor judge messages, show empty state -->
    {:else}
      <div class="empty-state"> 
        <i class="bi bi-gavel"></i> 
        <p>No results available yet</p>
      </div>
    {/if}
  </div>
</div>
{/if}
<style>
  .message-sidebar {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    background-color: #1a1a1a;
    border-left: 1px solid #333;
    max-width: 280px; /* Reduce sidebar width */
  }
  
  .sidebar-header {
    padding: 8px 12px;
    background-color: #2a2a2a;
    color: #ccc;
    border-bottom: 1px solid #444;
    flex-shrink: 0;
  }
  
  .sidebar-content {
    padding: 8px;
    overflow-y: auto;
    flex-grow: 1;
  }
  
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100px;
    margin: 20px 0;
    padding: 16px;
    color: #666;
    border: 1px dashed #444;
    border-radius: 4px;
    text-align: center;
  }
  
  .empty-state i {
    font-size: 24px;
    margin-bottom: 8px;
  }
  
  .summary-entry, .judge-entry {
    background-color: #252525;
    border-radius: 3px;
    border: 1px solid #383838;
    margin-bottom: 6px; /* Reduced from 10px */
    padding: 6px; /* Reduced from 8px */
    transition: background-color 0.2s ease;
  }
  
  .summary-entry:hover, .judge-entry:hover {
    background-color: #303030;
  }
  
  .summary-title {
    font-weight: bold;
    color: #00bcd4;
    font-size: 0.95em;
    margin-bottom: 6px;
    border-bottom: 1px dotted #444;
    padding-bottom: 4px;
  }
  
  .title-badge {
    margin-right: 4px;
  }
  
  .run-id {
    color: #888;
    font-size: 0.85em;
    font-weight: normal;
  }
  
  .summary-stats {
    margin-bottom: 6px;
    font-size: 0.85em;
  }
  
  .stat-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
  }
  
  .stat-label {
    color: #aaa;
  }
  
  .ascii-bar {
    margin-left: 4px;
    letter-spacing: -1px;
  }
  
  .predictions-list {
    font-size: 0.8em;
    margin-top: 6px;
    border-top: 1px dotted #444;
    padding-top: 4px;
  }
  
  .prediction-item {
    display: flex;
    align-items: center;
    margin-bottom: 1px; /* Reduced from 2px */
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }
  
  .pred-icon {
    width: 12px; /* Reduced from 16px */
    margin-right: 3px; /* Reduced from 4px */
  }
  
  .pred-agent {
    flex-grow: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-right: 4px; /* Reduced from 8px */
  }
  
  .pred-score {
    text-align: right;
    white-space: nowrap;
    font-size: 0.85em; /* Slightly smaller */
  }
  
  .no-predictions {
    color: #666;
    font-style: italic;
    text-align: center;
    padding: 8px;
  }
  
  .judge-info {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 4px;
  }
  
  .judge-name {
    font-weight: bold;
    color: #00bcd4;
    font-size: 0.9em;
  }
  
  .terminal-score {
    font-size: 0.9em;
    text-align: right;
    min-width: 60px;
    line-height: 1;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }
  
  .score-indicator {
    letter-spacing: -1px;
    display: inline-block;
    cursor: default;
  }
  
  .no-scores {
    color: #666;
    font-style: italic;
  }
  
  .judge-decision {
    display: flex;
    align-items: center;
    margin-bottom: 6px;
  }
  
  .decision-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    font-size: 0.9em;
    font-weight: bold;
    margin-right: 6px;
  }
  
  .decision-badge.violation {
    background-color: rgba(220, 53, 69, 0.2);
    color: #dc3545;
    border: 1px solid rgba(220, 53, 69, 0.4);
  }
  
  .decision-badge.clean {
    background-color: rgba(40, 167, 69, 0.2);
    color: #28a745;
    border: 1px solid rgba(40, 167, 69, 0.4);
  }
  
  .judge-conclusion {
    position: relative;
    color: #bbb;
    font-size: 0.85em;
    line-height: 1.3;
    margin: 8px 0;
    padding: 6px;
    background-color: #1e1e1e;
    border-radius: 3px;
    border-left: 2px solid #444;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: normal;
  }
  
  .toggle-btn {
    position: relative;
    padding: 0 3px;
    background: none;
    color: #777;
    border: none;
    font-size: 0.8em;
    cursor: pointer;
    margin-left: 5px;
  }
  
  .toggle-btn:hover {
    color: #ccc;
    background-color: #333;
    border-radius: 2px;
  }
  
  .action-buttons {
    display: flex;
    gap: 6px;
    margin: 5px 0;
  }
  
  .action-btn {
    display: inline-flex;
    align-items: center;
    background-color: #1e1e1e;
    color: #bbb;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 3px 8px;
    font-size: 0.75em;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .action-btn.active {
    background-color: #2a2a2a;
    border-color: #444;
    color: #eee;
  }
  
  .action-btn i {
    margin-right: 3px;
    font-size: 0.9em;
  }
  
  .action-btn:hover {
    background-color: #2a2a2a;
    color: #eee;
  }
  
  .reasons-btn.active {
    background-color: rgba(25, 135, 84, 0.15);
    border-color: rgba(25, 135, 84, 0.4);
  }
  
  .scores-btn.active {
    background-color: rgba(255, 193, 7, 0.15);
    border-color: rgba(255, 193, 7, 0.4);
  }
  
  .reasons-panel, .assessment-details {
    margin-top: 4px;
    margin-bottom: 8px;
    background-color: #1e1e1e;
    border-radius: 3px;
    padding: 6px;
    border-left: 2px solid #444;
  }
  
  .panel-title {
    font-size: 0.8em;
    color: #aaa;
    margin-bottom: 4px;
    font-weight: bold;
  }
  
  .reasons-list, .assessment-reasons {
    list-style-type: none;
    padding-left: 0;
    margin-bottom: 0;
    margin-top: 3px;
    font-size: 0.75em;
    color: #ddd;
  }
  
  .reasons-list li, .assessment-reasons li {
    padding: 3px 5px;
    position: relative;
    margin-bottom: 3px;
    border-left: 2px solid #444;
    padding-left: 8px;
    line-height: 1.3;
  }
  
  .assessment-section {
    margin-top: 8px;
    border-top: 1px dotted #444;
    padding-top: 5px;
  }
  
  .assessment-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8em;
    color: #aaa;
    margin-bottom: 3px;
  }
  
  .assessment-item {
    margin-bottom: 8px;
    padding-bottom: 5px;
    border-bottom: 1px dotted #333;
  }
  
  .assessment-item:last-child {
    margin-bottom: 0;
    border-bottom: none;
  }
  
  .assessment-item-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8em;
    margin-bottom: 3px;
  }
  
  .assessment-agent {
    font-weight: bold;
    color: #bbb;
  }
  
  .assessment-score {
    font-weight: bold;
  }
  
  /* Agent info & metadata styles */
  .judge-info {
    flex-direction: column;
    margin-bottom: 4px; /* Reduced from 8px */
  }
  
  .judge-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }
  
  .agent-metadata {
    display: flex;
    flex-wrap: wrap;
    gap: 3px; /* Reduced from 4px */
    margin-top: 1px; /* Reduced from 2px */
    margin-bottom: 2px; /* Reduced from 4px */
  }
  
  .metadata-badge {
    font-size: 0.7em;
    padding: 2px 5px;
    border-radius: 3px;
    display: inline-flex;
    align-items: center;
    gap: 3px;
    cursor: default;
  }
  
  .metadata-badge i {
    font-size: 0.9em;
  }
  
  .model-badge {
    background-color: rgba(13, 110, 253, 0.15);
    color: #0d6efd;
    border: 1px solid rgba(13, 110, 253, 0.3);
  }
  
  .template-badge {
    background-color: rgba(111, 66, 193, 0.15);
    color: #6f42c1;
    border: 1px solid rgba(111, 66, 193, 0.3);
  }
  
  .criteria-badge {
    background-color: rgba(32, 201, 151, 0.15);
    color: #20c997;
    border: 1px solid rgba(32, 201, 151, 0.3);
  }
  
  /* Score chips styles */
  .score-chips {
    display: flex;
    flex-direction: column;
    margin-top: 4px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }
  
  .score-chip {
    display: flex;
    align-items: center;
    background-color: #1e1e1e;
    border-radius: 3px; /* Reduced from 4px */
    padding: 3px 6px; /* Reduced from 4px 8px */
    cursor: pointer;
    transition: background-color 0.2s ease;
    gap: 4px; /* Reduced from 6px */
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    margin-bottom: 2px; /* Add small gap between items */
  }
  
  .score-chip:hover {
    background-color: #2a2a2a;
  }
  
  .terminal-score-chip {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  
  .expand-indicator {
    font-size: 0.7em;
    color: #777;
    margin-left: 3px;
  }
  
  .score-details {
    background-color: #1a1a1a;
    border-radius: 3px;
    border-left: 2px solid #444;
    padding: 6px;
    margin: 2px 0 6px 2px;
    font-size: 0.75em;
  }
  
  /* Custom tooltip styles that will be added via JavaScript */
  :global(.score-tooltip) {
    position: absolute;
    background-color: #2a2a2a;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 0.8rem;
    z-index: 1000;
    color: #eee;
    pointer-events: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    max-width: 250px;
  }
  
  :global(.score-list) {
    list-style-type: none;
    margin: 0;
    padding: 0;
  }
  
  :global(.scorer-name) {
    font-weight: bold;
    margin-right: 4px;
  }
</style>
