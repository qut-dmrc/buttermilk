<!-- 
OSB Message Component

Displays OSB (Oversight Board) analysis results with proper formatting
for policy violations, recommendations, and multi-agent responses.

Follows the retro terminal IRC-style design while presenting complex
OSB analysis data in a clear, readable format.
-->

<script lang="ts">
  import { getAgentStyle } from '$lib/utils/messageUtils';
  import type { Message } from '$lib/utils/messageUtils';

  export let message: Message;
  export let expanded = false;

  // Extract OSB-specific data from message content
  $: osbData = typeof message.content === 'object' ? message.content : JSON.parse(message.content || '{}');
  $: agentStyle = getAgentStyle('OSB_ANALYSIS');

  // Format confidence score as percentage
  $: confidencePercentage = Math.round((osbData.confidence_score || 0) * 100);
  
  // Format processing time
  $: processingTime = osbData.processing_time ? `${osbData.processing_time.toFixed(1)}s` : 'N/A';

  // Determine priority styling
  $: priorityClass = osbData.case_priority === 'critical' ? 'text-danger' : 
                    osbData.case_priority === 'high' ? 'text-warning' : 
                    osbData.case_priority === 'medium' ? 'text-info' : 'text-light';
</script>

<div class="message-wrapper mb-2">
  <!-- OSB Header -->
  <div class="message-header d-flex align-items-center">
    <span class="agent-indicator me-2" style="color: {agentStyle.color};">
      [{osbData.case_number || 'OSB-ANALYSIS'}]
    </span>
    <span class="text-muted small">
      {new Date(message.timestamp || Date.now()).toLocaleTimeString()}
    </span>
    {#if osbData.case_priority}
      <span class="badge {priorityClass} ms-2">
        {osbData.case_priority.toUpperCase()}
      </span>
    {/if}
    <span class="text-muted small ms-auto">
      Confidence: {confidencePercentage}% | Time: {processingTime}
    </span>
  </div>

  <!-- OSB Content -->
  <div class="message-content mt-2">
    <!-- Synthesis Summary -->
    {#if osbData.synthesis_summary}
      <div class="osb-synthesis border-start border-info ps-3 mb-3">
        <h6 class="text-info mb-2">📋 Executive Summary</h6>
        <p class="mb-0">{osbData.synthesis_summary}</p>
      </div>
    {/if}

    <!-- Policy Violations -->
    {#if osbData.policy_violations && osbData.policy_violations.length > 0}
      <div class="osb-violations border-start border-danger ps-3 mb-3">
        <h6 class="text-danger mb-2">⚠️ Policy Violations</h6>
        <ul class="list-unstyled mb-0">
          {#each osbData.policy_violations as violation}
            <li class="mb-1">• {violation}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- Recommendations -->
    {#if osbData.recommendations && osbData.recommendations.length > 0}
      <div class="osb-recommendations border-start border-success ps-3 mb-3">
        <h6 class="text-success mb-2">✅ Recommendations</h6>
        <ul class="list-unstyled mb-0">
          {#each osbData.recommendations as recommendation}
            <li class="mb-1">• {recommendation}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- Agent Responses (Expandable) -->
    {#if osbData.agent_responses}
      <div class="osb-agents mb-3">
        <button 
          class="btn btn-outline-light btn-sm mb-2" 
          on:click={() => expanded = !expanded}
        >
          {expanded ? '▼' : '▶'} Agent Analysis Details ({Object.keys(osbData.agent_responses).length} agents)
        </button>
        
        {#if expanded}
          <div class="agent-details">
            {#each Object.entries(osbData.agent_responses) as [agentName, agentResponse]}
              <div class="agent-card border border-secondary rounded p-2 mb-2">
                <div class="d-flex align-items-center mb-2">
                  <span class="agent-name text-warning fw-bold">
                    {agentName.toUpperCase()}
                  </span>
                  {#if agentResponse.confidence}
                    <span class="badge bg-secondary ms-2">
                      {Math.round(agentResponse.confidence * 100)}%
                    </span>
                  {/if}
                  {#if agentResponse.processing_time}
                    <span class="text-muted small ms-auto">
                      {agentResponse.processing_time.toFixed(1)}s
                    </span>
                  {/if}
                </div>
                
                <!-- Agent-specific content -->
                {#if agentResponse.findings}
                  <div class="agent-findings mb-2">
                    <strong>Findings:</strong> {agentResponse.findings}
                  </div>
                {/if}
                
                {#if agentResponse.analysis}
                  <div class="agent-analysis mb-2">
                    <strong>Analysis:</strong> {agentResponse.analysis}
                  </div>
                {/if}
                
                {#if agentResponse.validation}
                  <div class="agent-validation mb-2">
                    <strong>Validation:</strong> {agentResponse.validation}
                  </div>
                {/if}
                
                {#if agentResponse.related_themes && agentResponse.related_themes.length > 0}
                  <div class="agent-themes mb-2">
                    <strong>Themes:</strong> {agentResponse.related_themes.join(', ')}
                  </div>
                {/if}
                
                {#if agentResponse.sources && agentResponse.sources.length > 0}
                  <div class="agent-sources">
                    <small class="text-muted">
                      Sources: {agentResponse.sources.join(', ')}
                    </small>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Precedent Cases -->
    {#if osbData.precedent_cases && osbData.precedent_cases.length > 0}
      <div class="osb-precedents border-start border-warning ps-3 mb-3">
        <h6 class="text-warning mb-2">📚 Related Precedents</h6>
        <ul class="list-unstyled mb-0">
          {#each osbData.precedent_cases as precedent}
            <li class="mb-1">
              <code class="text-warning">{precedent}</code>
            </li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- Sources Consulted -->
    {#if osbData.sources_consulted && osbData.sources_consulted.length > 0}
      <div class="osb-sources">
        <details class="text-muted">
          <summary class="small">Sources Consulted ({osbData.sources_consulted.length})</summary>
          <ul class="list-unstyled mt-2 mb-0">
            {#each osbData.sources_consulted as source}
              <li class="small">• {source}</li>
            {/each}
          </ul>
        </details>
      </div>
    {/if}
  </div>
</div>

<style>
  .message-wrapper {
    font-family: 'Courier New', monospace;
    background-color: rgba(0, 20, 40, 0.3);
    border: 1px solid rgba(100, 150, 200, 0.2);
    border-radius: 4px;
    padding: 12px;
  }

  .agent-indicator {
    font-weight: bold;
    font-family: 'Courier New', monospace;
  }

  .osb-synthesis,
  .osb-violations,
  .osb-recommendations,
  .osb-precedents {
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    padding: 8px;
  }

  .agent-card {
    background-color: rgba(0, 0, 0, 0.3);
    font-size: 0.9em;
  }

  .agent-name {
    font-family: 'Courier New', monospace;
    letter-spacing: 0.5px;
  }

  details summary {
    cursor: pointer;
    user-select: none;
  }

  details summary:hover {
    color: #fff !important;
  }

  .btn-outline-light:hover {
    background-color: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.3);
  }
</style>