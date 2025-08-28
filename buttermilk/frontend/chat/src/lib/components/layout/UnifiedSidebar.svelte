<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import {
  	criteriaStore,
  	datasetsStore,
  	findMatchingSession,
  	flowChoices,
  	flowRunning,
  	initializeApp,
  	isDemoMode,
  	recordsStore,
  	selectedCriteria,
  	selectedDataset,
  	selectedFlow,
  	selectedRecord
  } from '$lib/stores/apiStore';
  import { runFlowAction } from '$lib/stores/terminalActionsStore';
  import { onMount } from 'svelte';
  
  $: isTerminalPage = $page.route.id === '/terminal' || $page.route.id === '/terminal/[sessionId]';
  $: isScorePage = $page.route.id?.startsWith('/score');
  
  // Track if a matching session is available for the current parameters
  let hasMatchingSession = false;

  // Check for matching session when parameters change
  $: if ($selectedFlow && $selectedDataset && $selectedRecord && $selectedCriteria) {
    checkForMatchingSession();
  } else {
    hasMatchingSession = false;
  }

  async function checkForMatchingSession() {
    try {
      const sessionId = await findMatchingSession($selectedFlow, $selectedDataset, $selectedRecord, $selectedCriteria);
      hasMatchingSession = !!sessionId;
    } catch (error) {
      console.error('Error checking for matching session:', error);
      hasMatchingSession = false;
    }
  }

  // For score pages, we need different logic for records
  let records: any[] = [];
  let loading = false;
  let error: string | null = null;
  
  // Set the flow and fetch records with scores for score pages
  async function loadRecordsForFlow(flow: string, dataset?: string) {
    if (flow && $flowChoices.data.includes(flow)) {
      selectedFlow.set(flow);
      
      if (isScorePage) {
        // For score pages, fetch records with scores
        try {
          loading = true;
          error = null;
          
          let url;
          if (dataset) {
            url = `/api/flows/${encodeURIComponent(flow)}/datasets/${encodeURIComponent(dataset)}/records?include_scores=true`;
          } else {
            // Try the first available dataset
            const datasetsResponse = await fetch(`/api/flows/${encodeURIComponent(flow)}/info`);
            if (datasetsResponse.ok) {
              const flowInfo = await datasetsResponse.json();
              if (flowInfo.datasets && flowInfo.datasets.length > 0) {
                dataset = flowInfo.datasets[0];
                selectedDataset.set(dataset);
                url = `/api/flows/${encodeURIComponent(flow)}/datasets/${encodeURIComponent(dataset)}/records?include_scores=true`;
              } else {
                throw new Error('No datasets available for this flow');
              }
            } else {
              throw new Error('Failed to fetch flow info');
            }
          }
          
          const response = await fetch(url);
          if (!response.ok) {
            throw new Error(`Failed to fetch records: ${response.statusText}`);
          }
          
          const data = await response.json();
          records = data;
        } catch (err) {
          error = err instanceof Error ? err.message : 'Failed to fetch records';
          records = [];
        } finally {
          loading = false;
        }
      }
    }
  }
  
  // Initialize app data (but only if not on a session page where it's already handled)
  onMount(() => {
    // Don't initialize if on a session page - the session page handles initialization
    if (!isTerminalPage || !$page.params.sessionId) {
      initializeApp();
    }
  });
  
  // For score pages: Don't auto-select flow, let user choose
  
  let initialDatasetSet = false;

  // Auto-select first dataset when datasets become available
  $: if ($datasetsStore.data.length > 0 && !$selectedDataset) {
    console.debug('Auto-selecting first dataset:', $datasetsStore.data[0]);
    selectedDataset.set($datasetsStore.data[0]);
    initialDatasetSet = true;
  }
  
  // For terminal pages: use the recordsStore
  $: if (isTerminalPage) {
    console.debug('Terminal page - full recordsStore:', $recordsStore);
    console.debug('Terminal page - recordsStore.data:', $recordsStore.data);
    console.debug('Terminal page - recordsStore.data type:', typeof $recordsStore.data);
    console.debug('Terminal page - recordsStore.data isArray:', Array.isArray($recordsStore.data));
    
    if ($recordsStore.data && Array.isArray($recordsStore.data)) {
      records = $recordsStore.data;
      console.debug('Terminal page - local records assigned:', records);
      console.debug('Terminal page - local records length:', records.length);
    } else if ($recordsStore.data?.records && Array.isArray($recordsStore.data.records)) {
      // Handle case where API returns {records: [...]} structure
      records = $recordsStore.data.records;
      console.debug('Terminal page - local records assigned from nested structure:', records);
      console.debug('Terminal page - local records length:', records.length);
    } else {
      records = [];
      console.debug('Terminal page - no valid records data, clearing records');
    }
  }
  
  // Handle flow change
  function handleFlowChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    const newFlow = target.value;
    selectedFlow.set(newFlow);
    
    if (isScorePage) {
      loadRecordsForFlow(newFlow);
    } else if (isTerminalPage && !$isDemoMode) {
      // Reset dataset and record when flow changes (but not in demo mode)
      selectedDataset.set('');
      selectedRecord.set('');
    }
  }
  
  // Handle dataset change
  function handleDatasetChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    const newDataset = target.value;
    console.log('Dataset changed to:', newDataset);
    selectedDataset.set(newDataset);
    
    if (isScorePage && $selectedFlow) {
      loadRecordsForFlow($selectedFlow, newDataset);
    } else if (isTerminalPage && !$isDemoMode) {
      console.debug('Terminal page - resetting record');
      // Reset record when dataset changes (but not in demo mode)
      selectedRecord.set('');
      // refetchRecords() is now handled by the store subscriber
    }
  }
  
  function runFlow() {
    flowRunning.set(true);
    $runFlowAction && $runFlowAction();
  }

  // Handle demo mode reload - find matching session and force page reload
  async function handleDemoReload() {
    try {
      console.log('Demo reload: Finding matching session for:', {
        flow: $selectedFlow,
        dataset: $selectedDataset,
        record: $selectedRecord,
        criteria: $selectedCriteria
      });

      const matchingSessionId = await findMatchingSession($selectedFlow, $selectedDataset, $selectedRecord, $selectedCriteria);
      
      if (matchingSessionId) {
        console.log(`Demo reload: Found session ${matchingSessionId}, navigating...`);
        // Navigate to the correct session using SvelteKit
        await goto(`/terminal/${matchingSessionId}`);
      } else {
        console.log('Demo reload: No matching session found');
        alert('No session found with the selected parameters');
      }
    } catch (error) {
      console.error('Demo reload error:', error);
      alert('Failed to find matching session');
    }
  }
</script>

{#if isTerminalPage}
	<!-- Terminal API Selector Section -->
	<div class="terminal-selector">
		<h4 class="terminal-title">API Selector</h4>

		<!-- Flow Dropdown -->
		<div class="selector-group">
			<label for="flow-select" class="form-label">Select Flow:</label>
			{#if $flowChoices.loading}
				<div class="terminal-loading">Loading flows...</div>
			{:else if $flowChoices.error}
				<div class="terminal-error">Error: {$flowChoices.error}</div>
			{:else if $flowChoices.data.length > 0}
				<select
					id="flow-select"
					class="form-select terminal-select"
					bind:value={$selectedFlow}
					onchange={handleFlowChange}
				>
					<option value="">Choose a flow...</option>
					{#each $flowChoices.data as flow}
						<option value={flow}>{flow.toUpperCase()}</option>
					{/each}
				</select>
			{:else}
				<div class="terminal-warning">No flows available</div>
			{/if}
		</div>

		<!-- Dataset Dropdown -->
		<div class="selector-group">
			<label for="dataset-select" class="form-label">Select Dataset:</label>
			{#if !$selectedFlow}
				<select class="form-select terminal-select" disabled>
					<option>Choose a flow first...</option>
				</select>
			{:else if $datasetsStore.loading}
				<div class="terminal-loading">Loading datasets...</div>
			{:else if $datasetsStore.error}
				<div class="terminal-error">Error: {$datasetsStore.error}</div>
			{:else if $datasetsStore.data.length > 0}
				<select
					id="dataset-select"
					class="form-select terminal-select"
					bind:value={$selectedDataset}
					onchange={handleDatasetChange}
				>
					<option value="">Choose a dataset...</option>
					{#each $datasetsStore.data as dataset}
						<option value={dataset}>{dataset.toUpperCase()}</option>
					{/each}
				</select>
			{:else}
				<div class="terminal-warning">No datasets available</div>
			{/if}
		</div>

		<!-- Record Dropdown -->
		<div class="selector-group">
			<label for="record-select" class="form-label">Select Record:</label>
			{#if !$selectedFlow || !$selectedDataset}
				<select class="form-select terminal-select" disabled>
					<option>Choose flow and dataset first...</option>
				</select>
			{:else if records.length === 0}
				<div class="terminal-warning">No records available</div>
			{:else}
				<select id="record-select" class="form-select terminal-select" bind:value={$selectedRecord}>
					<option value="">Choose a record...</option>
					{#each records as record}
						<option value={record.record_id}>{record.name || record.record_id}</option>
					{/each}
				</select>
			{/if}
		</div>

		<!-- Criteria Dropdown -->
		<div class="selector-group">
			<label for="criteria-select" class="form-label">Select Criteria:</label>
			{#if !$selectedFlow}
				<select class="form-select terminal-select" disabled>
					<option>Choose a flow first...</option>
				</select>
			{:else if $criteriaStore.data.length > 0}
				<select
					id="criteria-select"
					class="form-select terminal-select"
					bind:value={$selectedCriteria}
				>
					<option value="">Choose criteria...</option>
					{#each $criteriaStore.data as criteria}
						<option value={criteria}>{criteria}</option>
					{/each}
				</select>
			{:else}
				<div class="terminal-warning">No criteria available</div>
			{/if}
		</div>

		<!-- Action Buttons -->
		{#if $selectedFlow }
			<div class="run-button-container">
				{#if hasMatchingSession}
					<button class="btn load-session-button" onclick={handleDemoReload}>
						LOAD SESSION
					</button>
				{/if}
				{#if !$isDemoMode}
					<!-- Only show Run Flow in live mode -->
					<button class="btn terminal-button" onclick={runFlow}>
						RUN FLOW
					</button>
				{/if}
			</div>
		{/if}
	</div>
{:else if isScorePage}
	<!-- Score Page Sidebar -->
	<div class="score-sidebar">
		<h4 class="terminal-title">
			{#if $selectedFlow}
				{$selectedFlow.toUpperCase()} Scores
			{:else}
				Score Analysis
			{/if}
		</h4>

		<!-- Flow selection -->
		<div class="selector-group">
			<label for="flow-select" class="form-label">Select Flow:</label>
			{#if $flowChoices.loading}
				<div class="terminal-loading">Loading flows...</div>
			{:else if $flowChoices.error}
				<div class="terminal-error">Error: {$flowChoices.error}</div>
			{:else if $flowChoices.data.length > 0}
				<select
					id="flow-select"
					class="form-select terminal-select"
					bind:value={$selectedFlow}
					onchange={handleFlowChange}
				>
					<option value="">Choose a flow...</option>
					{#each $flowChoices.data as flow}
						<option value={flow}>{flow.toUpperCase()}</option>
					{/each}
				</select>
			{:else}
				<div class="terminal-warning">No flows available</div>
			{/if}
		</div>

		<!-- Dataset selection for score pages -->
		{#if $datasetsStore.data.length > 1}
			<div class="selector-group">
				<label for="dataset-select" class="form-label">Dataset:</label>
				<select
					id="dataset-select"
					class="form-select terminal-select"
					bind:value={$selectedDataset}
					onchange={handleDatasetChange}
				>
					{#each $datasetsStore.data as dataset}
						<option value={dataset}>{dataset.toUpperCase()}</option>
					{/each}
				</select>
			</div>
		{/if}

		<!-- Records list -->
		<div class="records-list">
			<h5 class="sidebar-section-title">Records</h5>

			{#if !$selectedFlow}
				<div class="terminal-warning">Select a flow to view records</div>
			{:else if loading}
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
								href="/score/{encodeURIComponent($selectedFlow)}/{encodeURIComponent(
									$selectedDataset || 'default'
								)}/{record.record_id}"
								class="record-link"
								class:active={$page.params.record_id === record.record_id}
							>
								<div class="record-header">
									<span class="record-id">{record.record_id}</span>
									<span class="record-name">{record.name}</span>
								</div>
								{#if record.summary_scores}
									<div class="record-scores">
										<div class="score-item">
											<span class="score-label">Off-shelf:</span>
											<span class="score-value"
												>{(record.summary_scores.off_shelf_accuracy * 100).toFixed(0)}%</span
											>
										</div>
										<div class="score-item">
											<span class="score-label">Custom:</span>
											<span class="score-value"
												>{(record.summary_scores.custom_average * 100).toFixed(0)}%</span
											>
										</div>
									</div>
								{/if}
							</a>
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	</div>
{:else}
	<!-- Default/other pages sidebar -->
	<div class="default-sidebar">
		<div class="alert alert-secondary mt-5 col-sm-7" role="alert">
			Warning: explicit content and hateful ideologies here!
		</div>

		<div>
			<h5 class="header">Drag Queens vs White Supremacists examples:</h5>
			<ul>
				<li><a href="/score/drag/example1">Example Drag Link 1</a></li>
				<li><a href="/score/drag/example2">Example Drag Link 2</a></li>
			</ul>
		</div>

		<div>
			<h5 class="header">Oversight Board examples:</h5>
			<ul>
				<li><a href="/score/osb/example1">Example OSB Link 1</a></li>
				<li><a href="/score/osb/example2">Example OSB Link 2</a></li>
			</ul>
		</div>

		<div>
			<h5 class="header">Tone Policing examples:</h5>
			<ul>
				<li><a href="/score/tonepolice/example1">Example Tone Link 1</a></li>
				<li><a href="/score/tonepolice/example2">Example Tone Link 2</a></li>
			</ul>
		</div>
	</div>
{/if}

<style>
	.terminal-selector,
	.score-sidebar {
		padding: 1rem;
		
		min-height: 80vh;
	}

	.score-sidebar {
		background-color: rgba(0, 0, 0, 0.1);
		border-right: 1px solid rgba(255, 255, 255, 0.1);
	}

	.terminal-title {
		font-weight: bold;
		margin-bottom: 1rem;
		border-bottom: 1px solid rgba(255, 255, 255, 0.2);
		padding-bottom: 0.5rem;
		font-size: 1.2rem;
	}

	.selector-group {
		margin-bottom: 1rem;
	}

	.form-label {
		color: #00ffff;
		font-size: 0.8rem;
		margin-bottom: 0.5rem;
		text-transform: uppercase;
		font-weight: bold;
	}

	.terminal-select {
		background-color: rgba(0, 0, 0, 0.5);
		border: 1px solid rgba(255, 255, 255, 0.3);
		color: #fff;
		
		font-size: 0.9rem;
	}

	.terminal-select:focus {
		background-color: rgba(0, 0, 0, 0.7);
		border-color: #00ff00;
		box-shadow: 0 0 0 0.25rem rgba(0, 255, 0, 0.25);
		color: #fff;
	}

	.terminal-select:disabled {
		background-color: rgba(0, 0, 0, 0.3);
		border-color: rgba(255, 255, 255, 0.1);
		color: #666;
	}

	.terminal-button {
		background-color: rgba(0, 255, 0, 0.1);
		border: 1px solid #00ff00;
		color: #00ff00;
		
		font-weight: bold;
		text-transform: uppercase;
		width: 100%;
		padding: 0.75rem;
		transition: all 0.2s ease;
	}

	.terminal-button:hover {
		background-color: rgba(0, 255, 0, 0.2);
		box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
	}

	.run-button-container {
		margin-top: 1.5rem;
	}

	.sidebar-section-title {
		color: #00ffff;
		font-size: 1rem;
		margin-bottom: 0.5rem;
		text-transform: uppercase;
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

	.record-header {
		margin-bottom: 0.25rem;
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

	.record-scores {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.25rem;
		font-size: 0.8rem;
	}

	.score-item {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}

	.score-label {
		color: #aaa;
	}

	.score-value {
		color: #00ffff;
		font-weight: bold;
	}

	.terminal-loading,
	.terminal-error,
	.terminal-warning {
		padding: 0.5rem;
		font-size: 0.9rem;
		text-align: center;
		border-radius: 4px;
		margin-bottom: 0.5rem;
	}

	.terminal-loading {
		color: #00ffff;
		background-color: rgba(0, 255, 255, 0.1);
		border: 1px solid rgba(0, 255, 255, 0.3);
	}

	.terminal-error {
		color: #ff4444;
		background-color: rgba(255, 68, 68, 0.1);
		border: 1px solid rgba(255, 68, 68, 0.3);
	}

	.terminal-warning {
		color: #ffaa00;
		background-color: rgba(255, 170, 0, 0.1);
		border: 1px solid rgba(255, 170, 0, 0.3);
	}

	.default-sidebar {
		padding: 1rem;
	}

	.header {
		color: #00ffff;
		margin-top: 1.5rem;
		margin-bottom: 0.5rem;
	}

	/* Load Session Button (Orange styling) */
	.load-session-button {
		background-color: #fd7e14;
		color: white;
		border: none;
		padding: 0.5rem 1rem;
		margin-top: 0.5rem;
		font-size: 0.875rem;
		font-weight: bold;
		border-radius: 4px;
		cursor: pointer;
		transition: background-color 0.2s;
		width: 100%;
	}

	.load-session-button:hover {
		background-color: #e85a00;
	}

	.load-session-button:active {
		background-color: #d04701;
	}

	/* Ensure both buttons are properly spaced */
	.run-button-container .btn + .btn {
		margin-top: 0.5rem;
	}
</style>
