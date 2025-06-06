<script lang="ts">
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import { selectedFlow, recordsStore, initializeApp, flowChoices } from '$lib/stores/apiStore';
  import { browser } from '$app/environment';

  let currentFlow = ''; // will be set from available flows
  let records: any[] = [];
  let loading = false;
  let error: string | null = null;

  // Set the flow and fetch records
  function loadRecordsForFlow(flow: string) {
    if (flow && $flowChoices.data.includes(flow)) {
      selectedFlow.set(flow);
      currentFlow = flow;
    }
  }

  // Set default flow when flow choices are loaded
  $: if ($flowChoices.data.length > 0 && !currentFlow) {
    currentFlow = $flowChoices.data[0];
    loadRecordsForFlow(currentFlow);
  }

  // Subscribe to records store
  $: {
    if ($recordsStore) {
      records = $recordsStore.data;
      loading = $recordsStore.loading;
      error = $recordsStore.error;
    }
  }

  onMount(() => {
    if (browser) {
      initializeApp();
    }
  });

  // Handle flow change
  function handleFlowChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    loadRecordsForFlow(target.value);
  }
</script>

<div class="score-layout">
  <div class="row">
    <!-- Sidebar for record selection -->
    <div class="col-md-3 score-sidebar">
      <div class="sidebar-header">
        <h4 class="terminal-title">Toxicity Scores</h4>
        
        <!-- Flow selection -->
        <div class="flow-selector mb-3">
          <label for="flow-select" class="form-label">Dataset:</label>
          {#if $flowChoices.loading}
            <div class="terminal-loading">Loading flows...</div>
          {:else if $flowChoices.error}
            <div class="terminal-error">Error: {$flowChoices.error}</div>
          {:else if $flowChoices.data.length > 0}
            <select 
              id="flow-select" 
              class="form-select terminal-select" 
              bind:value={currentFlow}
              on:change={handleFlowChange}
            >
              {#each $flowChoices.data as flow}
                <option value={flow}>{flow.toUpperCase()}</option>
              {/each}
            </select>
          {:else}
            <div class="terminal-warning">No flows available</div>
          {/if}
        </div>

        <!-- Records list -->
        <div class="records-list">
          <h5 class="sidebar-section-title">Records</h5>
          
          {#if loading}
            <div class="terminal-loading">Loading records...</div>
          {:else if error}
            <div class="terminal-error">Error: {error}</div>
          {:else if records.length === 0}
            <div class="terminal-warning">No records found</div>
          {:else}
            <ul class="record-list">
              {#each records as record}
                <li class="record-item">
                  <a 
                    href="/score/{record.id}" 
                    class="record-link"
                    class:active={$page.params.record_id === record.id}
                  >
                    <span class="record-id">{record.id}</span>
                    <span class="record-name">{record.name}</span>
                  </a>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      </div>
    </div>

    <!-- Main content area -->
    <div class="col-md-9 score-content">
      <slot></slot>
    </div>
  </div>
</div>

<style>
  .score-layout {
    padding: 1rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }

  .score-sidebar {
    background-color: rgba(0, 0, 0, 0.1);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1rem;
    min-height: 80vh;
  }

  .terminal-title {
    color: #00ff00;
    font-weight: bold;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    padding-bottom: 0.5rem;
  }

  .sidebar-section-title {
    color: #00ffff;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
  }

  .terminal-select {
    background-color: rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: #fff;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }

  .terminal-select:focus {
    background-color: rgba(0, 0, 0, 0.7);
    border-color: #00ff00;
    box-shadow: 0 0 0 0.25rem rgba(0, 255, 0, 0.25);
  }

  .record-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .record-item {
    margin-bottom: 0.25rem;
  }

  .record-link {
    display: block;
    padding: 0.5rem;
    text-decoration: none;
    color: #ccc;
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid transparent;
    transition: all 0.2s ease;
  }

  .record-link:hover {
    background-color: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.2);
    color: #fff;
  }

  .record-link.active {
    background-color: rgba(0, 255, 0, 0.2);
    border-color: #00ff00;
    color: #00ff00;
  }

  .record-id {
    display: block;
    font-size: 0.85rem;
    color: #888;
  }

  .record-name {
    display: block;
    font-size: 0.9rem;
    font-weight: bold;
  }

  .terminal-loading,
  .terminal-error,
  .terminal-warning {
    padding: 0.5rem;
    font-size: 0.9rem;
    text-align: center;
  }

  .terminal-loading {
    color: #00ffff;
  }

  .terminal-error {
    color: #ff4444;
  }

  .terminal-warning {
    color: #ffaa00;
  }

  .score-content {
    padding: 1rem;
  }
</style>