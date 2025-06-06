<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  import { recordsStore, selectedFlow, flowChoices, initializeApp } from '$lib/stores/apiStore';

  let records: any[] = [];

  $: {
    if ($recordsStore) {
      records = $recordsStore.data;
    }
  }

  onMount(() => {
    // Initialize app to load flow choices
    initializeApp();
    
    // If we have records, redirect to the first one
    if (records.length > 0) {
      goto(`/score/${records[0].id}`);
    }
  });

  // Watch for records changes and auto-redirect
  $: if (records.length > 0) {
    goto(`/score/${records[0].id}`);
  }
</script>

<div class="score-home">
  <div class="terminal-header">
    <h1 class="terminal-title">Toxicity Score Analysis</h1>
    <div class="terminal-subtitle">Select a record from the sidebar to view detailed scoring results</div>
  </div>

  <div class="terminal-content">
    <div class="info-panel">
      <h3 class="panel-title">Available Datasets</h3>
      {#if $flowChoices.loading}
        <div class="loading-text">Loading available flows...</div>
      {:else if $flowChoices.error}
        <div class="error-text">Error loading flows: {$flowChoices.error}</div>
      {:else if $flowChoices.data.length > 0}
        <ul class="dataset-list">
          {#each $flowChoices.data as flow}
            <li><strong>{flow.toUpperCase()}</strong> - {flow} dataset</li>
          {/each}
        </ul>
      {:else}
        <div class="no-data-text">No datasets available</div>
      {/if}
    </div>

    <div class="info-panel">
      <h3 class="panel-title">Score Interpretation</h3>
      <div class="score-legend">
        <div class="score-item">
          <span class="score-indicator high">█████</span>
          <span class="score-text">High toxicity (80-100%)</span>
        </div>
        <div class="score-item">
          <span class="score-indicator medium">███░░</span>
          <span class="score-text">Medium toxicity (40-79%)</span>
        </div>
        <div class="score-item">
          <span class="score-indicator low">█░░░░</span>
          <span class="score-text">Low toxicity (0-39%)</span>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .score-home {
    padding: 2rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }

  .terminal-header {
    text-align: center;
    margin-bottom: 3rem;
    border-bottom: 2px solid rgba(0, 255, 0, 0.3);
    padding-bottom: 1rem;
  }

  .terminal-title {
    color: #00ff00;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
  }

  .terminal-subtitle {
    color: #00ffff;
    font-size: 1.2rem;
  }

  .terminal-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 2rem;
  }

  .info-panel {
    background-color: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 1.5rem;
    border-radius: 4px;
  }

  .panel-title {
    color: #00ffff;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 0.5rem;
  }

  .dataset-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .dataset-list li {
    padding: 0.5rem 0;
    color: #ccc;
    border-bottom: 1px dotted rgba(255, 255, 255, 0.1);
  }

  .dataset-list strong {
    color: #00ff00;
    display: inline-block;
    width: 120px;
  }

  .score-legend {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .score-item {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .score-indicator {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 1.2rem;
    letter-spacing: -1px;
  }

  .score-indicator.high {
    color: #ff4444;
  }

  .score-indicator.medium {
    color: #ffaa00;
  }

  .score-indicator.low {
    color: #00ff00;
  }

  .score-text {
    color: #ccc;
  }

  .loading-text {
    color: #ffaa00;
    font-style: italic;
    padding: 0.5rem 0;
  }

  .error-text {
    color: #ff4444;
    padding: 0.5rem 0;
  }

  .no-data-text {
    color: #666;
    font-style: italic;
    padding: 0.5rem 0;
  }

  @media (max-width: 768px) {
    .terminal-content {
      grid-template-columns: 1fr;
    }
  }
</style>