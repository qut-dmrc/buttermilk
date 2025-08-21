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

		// If we have records and a selected flow, redirect to the first one
		if (records.length > 0 && $selectedFlow) {
			goto(`/score/${encodeURIComponent($selectedFlow)}/${records[0].id}`);
		}
	});

	// Watch for records changes and auto-redirect
	$: if (records.length > 0 && $selectedFlow) {
		goto(`/score/${encodeURIComponent($selectedFlow)}/${records[0].id}`);
	}
</script>

<div class="score-home">
	<div class="terminal-header">
		<h1 class="terminal-title">
			{#if $selectedFlow}
				{$selectedFlow.toUpperCase()} Score Analysis
			{:else}
				Score Analysis
			{/if}
		</h1>
		<div class="terminal-subtitle">
			{#if $selectedFlow}
				Select a record from the sidebar to view detailed scoring results
			{:else}
				Select a flow from the sidebar to begin analysis
			{/if}
		</div>
	</div>

	<div class="terminal-content">
		<div class="info-panel">
			<h3 class="panel-title">Available Flows</h3>
			{#if $flowChoices.loading}
				<div class="loading-text">Loading available flows...</div>
			{:else if $flowChoices.error}
				<div class="error-text">Error loading flows: {$flowChoices.error}</div>
			{:else if $flowChoices.data.length > 0}
				<ul class="dataset-list">
					{#each $flowChoices.data as flow}
						<li>
							<strong>{flow.toUpperCase()}</strong> - {flow} analysis flow
							{#if flow === $selectedFlow}
								<span class="selected-indicator">← Selected</span>
							{/if}
						</li>
					{/each}
				</ul>
			{:else}
				<div class="no-data-text">No flows available</div>
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
