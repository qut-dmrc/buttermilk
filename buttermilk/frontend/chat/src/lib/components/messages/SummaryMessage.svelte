<script lang="ts">
  import { onMount } from 'svelte';
  import { slide } from 'svelte/transition';
  import { getAgentStyle, getAgentEmoji, getScoreColor, type Message, type SummaryResult } from '$lib/utils/messageUtils';

  // Props
  export let message: Message;
  export let expanded = false;

  // Local state
  let isContentVisible = false;
  let tooltipElements: HTMLElement[] = [];
  let Tooltip: any;

  // Lifecycle
  onMount(() => {
    // Only load Bootstrap in browser environment
    if (typeof window !== 'undefined') {
      import('bootstrap').then(bootstrap => {
        Tooltip = bootstrap.Tooltip;
        initTooltips();
      }).catch(e => {
        console.warn('Bootstrap not available:', e);
      });
    }

    return () => {
      // Destroy tooltips on component unmount
      if (Tooltip && tooltipElements.length > 0) {
        tooltipElements.forEach(el => {
          const tooltip = Tooltip.getInstance(el);
          if (tooltip) tooltip.dispose();
        });
      }
    };
  });

  // Initialize Bootstrap tooltips
  function initTooltips() {
    if (!Tooltip || typeof document === 'undefined') return;
    
    // Small delay to ensure DOM is ready
    setTimeout(() => {
      tooltipElements = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
      tooltipElements.forEach(el => {
        try {
          new Tooltip(el, {
            html: true,
            trigger: 'hover'
          });
        } catch (e) {
          console.warn('Error initializing tooltip:', e);
        }
      });
    }, 100);
  }

  // Get agent styling
  $: agentStyle = getAgentStyle(message.agent_info?.agent_name || 'System');
  $: agentEmoji = getAgentEmoji(message.agent_info?.agent_name || 'System', message.agent_info?.agent_id || '');
  
  // Extract summary data from message outputs
  $: summaryData = message.outputs as SummaryResult;

  // Toggle content visibility
  function toggleContent() {
    isContentVisible = !isContentVisible;
  }

  // Format predictions for display
  function formatPredictions(predictions: any[]) {
    if (!predictions || predictions.length === 0) return "No predictions available";
    
    return predictions.map(pred => {
      const predictionIcon = pred.prediction ? "✅" : "❌";
      const scoreColor = pred.score ? getScoreColor(pred.score) : "#6c757d";
      const scoreText = pred.score ? `${(pred.score * 100).toFixed(0)}%` : "N/A";
      
      return `${predictionIcon} ${pred.agent_name || pred.agent_id}: ${scoreText}`;
    }).join("\n");
  }

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

  // Count predictions by result (true/false)
  function countPredictions(predictions: any[]) {
    if (!predictions || predictions.length === 0) {
      return { positive: 0, negative: 0, total: 0 };
    }
    
    const counts = predictions.reduce((acc, pred) => {
      if (pred.prediction === true) {
        acc.positive += 1;
      } else {
        acc.negative += 1;
      }
      acc.total += 1;
      return acc;
    }, { positive: 0, negative: 0, total: 0 });
    
    return counts;
  }

  // Calculate agreement rate as a string
  function calculateAgreementRate(predictions: any[]): string {
    if (!predictions || predictions.length <= 1) return "N/A";
    
    const counts = countPredictions(predictions);
    const max = Math.max(counts.positive, counts.negative);
    const agreementRate = counts.total > 0 ? max / counts.total : 0;
    
    return (agreementRate * 100).toFixed(0) + "%";
  }

  // Get majority decision
  function getMajorityDecision(predictions: any[]): string {
    if (!predictions || predictions.length === 0) return "Unknown";
    
    const counts = countPredictions(predictions);
    
    if (counts.positive > counts.negative) {
      return "✅ Violates";
    } else if (counts.negative > counts.positive) {
      return "❌ Does Not Violate";
    } else {
      return "⚖️ Tied";
    }
  }
</script>

<div class="message-terminal">
  <span class="timestamp">[{message.timestamp}]</span>
  <span class="agent-name" style="color: {agentStyle.color}">
    {agentEmoji} {message.agent_info?.agent_name}:
  </span>
  
  <span class="message-content summary-message">
    <!-- Title -->
    <div class="summary-title">
      Run Summary {summaryData.run_id ? `#${summaryData.run_id.substring(0, 6)}` : ''}
    </div>
    
    <!-- Compact Summary -->
    <div class="summary-header">
      <div class="summary-stat" data-bs-toggle="tooltip" data-bs-title="Majority Decision">
        <span class="stat-label">Decision:</span> 
        <span class="stat-value">{getMajorityDecision(summaryData.predictions)}</span>
      </div>
      <div class="summary-stat" data-bs-toggle="tooltip" data-bs-title="Judge Agreement Rate">
        <span class="stat-label">Agreement:</span> 
        <span class="stat-value">{summaryData.agreement_rate ? `${(summaryData.agreement_rate * 100).toFixed(0)}%` : calculateAgreementRate(summaryData.predictions)}</span>
      </div>
      <div class="summary-stat" data-bs-toggle="tooltip" data-bs-title="Average Judge Score">
        <span class="stat-label">Avg Score:</span> 
        <span class="stat-value" style="color: {getScoreColor(summaryData.avg_score)}">
          {summaryData.avg_score ? `${(summaryData.avg_score * 100).toFixed(0)}%` : 'N/A'}
          <span class="ascii-bar">{generateScoreBar(summaryData.avg_score)}</span>
        </span>
      </div>
    </div>
    
    <!-- Toggle Button -->
    <div class="content-toggle">
      <button class="terminal-button" on:click={toggleContent}>
        {isContentVisible ? '[-]' : '[+]'} details
      </button>
    </div>
    
    <!-- Detailed Content -->
    {#if isContentVisible}
      <div class="summary-details" transition:slide={{ duration: 200 }}>
        <div class="prediction-list">
          <div class="prediction-header">
            <span class="pred-agent">Judge</span>
            <span class="pred-decision">Decision</span>
            <span class="pred-uncertainty">Uncertainty</span>
            <span class="pred-score">Score</span>
          </div>
          
          {#each summaryData.predictions as prediction}
            {@const predictionColor = prediction.prediction ? "#5cb85c" : "#dc3545"}
            {@const scoreColor = getScoreColor(prediction.score)}
            
            <div class="prediction-item">
              <span class="pred-agent">{prediction.agent_name || prediction.agent_id}</span>
              <span class="pred-decision" style="color: {predictionColor}">
                {prediction.prediction ? "Violates" : "Does Not Violate"}
              </span>
              <span class="pred-uncertainty">{prediction.uncertainty || "N/A"}</span>
              <span class="pred-score" style="color: {scoreColor}">
                {prediction.score ? `${(prediction.score * 100).toFixed(0)}%` : "N/A"}
                <span class="ascii-bar">{generateScoreBar(prediction.score)}</span>
              </span>
            </div>
            
            {#if prediction.assessments && prediction.assessments.length > 0}
              <div class="assessment-list">
                {#each prediction.assessments as assessment}
                  {@const assessScoreColor = getScoreColor(assessment.score)}
                  <div class="assessment-item">
                    <span class="assess-agent">{assessment.agent_name || assessment.agent_id}</span>
                    <span class="assess-score" style="color: {assessScoreColor}">
                      {(assessment.score * 100).toFixed(0)}%
                      <span class="ascii-bar">{generateScoreBar(assessment.score)}</span>
                    </span>
                    {#if assessment.text}
                      <span class="assess-text">{assessment.text}</span>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}
          {/each}
        </div>
      </div>
    {/if}
  </span>
</div>

<style>
  .summary-message {
    display: block;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    padding: 8px;
    margin-top: 5px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }
  
  .summary-title {
    font-size: 1em;
    font-weight: bold;
    margin-bottom: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 4px;
  }
  
  .summary-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 0.9em;
  }
  
  .summary-stat {
    padding: 0 4px;
  }
  
  .stat-label {
    color: #aaa;
    margin-right: 4px;
  }
  
  .ascii-bar {
    margin-left: 4px;
    letter-spacing: -1px;
  }
  
  .terminal-button {
    background: none;
    border: none;
    color: #aaa;
    cursor: pointer;
    padding: 2px 4px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 0.85em;
    transition: color 0.1s ease;
  }
  
  .terminal-button:hover {
    color: #fff;
  }
  
  .summary-details {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed rgba(255, 255, 255, 0.1);
  }
  
  .prediction-list {
    font-size: 0.85em;
  }
  
  .prediction-header {
    color: #aaa;
    margin-bottom: 4px;
    display: grid;
    grid-template-columns: 25% 25% 25% 25%;
    padding: 2px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .prediction-item {
    display: grid;
    grid-template-columns: 25% 25% 25% 25%;
    padding: 3px 0;
    border-bottom: 1px dotted rgba(255, 255, 255, 0.05);
  }
  
  .assessment-list {
    padding-left: 20px;
    margin-bottom: 6px;
  }
  
  .assessment-item {
    display: grid;
    grid-template-columns: 25% 25% 50%;
    font-size: 0.85em;
    color: #bbb;
    padding: 2px 0;
  }
  
  .assess-text {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Import slide transition from svelte/transition */
  [transition\:slide] {
    transition: all 200ms ease-in-out;
  }
</style>
