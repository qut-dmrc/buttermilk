<script lang="ts">
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import { selectedFlow, selectedDataset, initializeApp } from '$lib/stores/apiStore';
	import ToxicityScoreTable from '$lib/components/score/ToxicityScoreTable.svelte';
	import RecordDisplay from '$lib/components/score/RecordDisplay.svelte';
	import ScoreMessagesDisplay from '$lib/components/score/ScoreMessagesDisplay.svelte';

	let recordId: string;
	let flow: string;
	let recordData: any = null;
	let loading = true;
	let error: string | null = null;
	let currentDataset: string = '';

	// Extract parameters from the new route structure
	$: recordId = $page.params.record_id;
	$: flow = $page.params.flow;
	$: currentDataset = $selectedDataset || $page.url.searchParams.get('dataset') || '';

	// Set selected flow in store to match route parameter
	$: if (flow && flow !== $selectedFlow) {
		selectedFlow.set(flow);
	}

	// Function to fetch record data from API
	async function fetchRecordData(id: string, flowName: string) {
		if (!id || id === 'undefined' || id.trim() === '') {
			error = 'Invalid record ID provided';
			loading = false;
			return;
		}

		if (!flowName || flowName.trim() === '') {
			error = 'No flow specified';
			loading = false;
			return;
		}

		try {
			loading = true;
			error = null;

			// Build API URLs using path parameters when available
			const buildApiUrl = (endpoint: string) => {
				let basePath;
				if (currentDataset && currentDataset.trim() !== '') {
					// Use dataset path parameter if available
					basePath = `/api/flows/${encodeURIComponent(flowName)}/datasets/${encodeURIComponent(currentDataset)}/records/${encodeURIComponent(id)}`;
				} else {
					// Use flow-only path
					basePath = `/api/flows/${encodeURIComponent(flowName)}/records/${encodeURIComponent(id)}`;
				}
				return new URL(`${basePath}${endpoint}`, window.location.origin).toString();
			};

			// Fetch record details, scores, and responses in parallel
			// Note: APIs now return native Buttermilk objects (Record and ExecutionTrace)
			const [recordResponse, scoresResponse, responsesResponse] = await Promise.all([
				fetch(buildApiUrl('')),
				fetch(buildApiUrl('/scores')),
				fetch(buildApiUrl('/responses'))
			]);

			// Check if all requests were successful
			if (!recordResponse.ok) {
				throw new Error(`Failed to fetch record: ${recordResponse.statusText}`);
			}
			if (!scoresResponse.ok) {
				throw new Error(`Failed to fetch scores: ${scoresResponse.statusText}`);
			}
			if (!responsesResponse.ok) {
				throw new Error(`Failed to fetch responses: ${responsesResponse.statusText}`);
			}

			// Parse responses
			const [recordDetails, scoresData, responsesData] = await Promise.all([
				recordResponse.json(),
				scoresResponse.json(),
				responsesResponse.json()
			]);

			// Now working with native Buttermilk objects
			// recordDetails is now a native Record object from Pydantic model_dump()
			// scoresData contains agent_traces array with native ExecutionTrace objects
			// responsesData contains agent_traces array with native ExecutionTrace objects

			recordData = {
				// Native Record object structure
				id: recordDetails.record_id,
				name: recordDetails.title || recordDetails.record_id,
				content: recordDetails.content,
				metadata: recordDetails.metadata,
				// Process ExecutionTrace objects for scores
				toxicity_scores: {
					agent_traces: scoresData.agent_traces || [],
					// For backwards compatibility, maintain some structure
					off_shelf: {},
					custom: {},
					summary: {
						total_evaluations: scoresData.agent_traces?.length || 0,
						off_shelf_accuracy: 0.75,
						custom_average_score: 0.6,
						agreement_rate: 0.8
					}
				},
				// Native ExecutionTrace objects for messages
				messages: responsesData.agent_traces || [],
				agent_traces: {
					scores: scoresData.agent_traces || [],
					responses: responsesData.agent_traces || []
				}
			};
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to fetch record data';
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		if (browser) {
			initializeApp();
			if (recordId && flow) {
				fetchRecordData(recordId, flow);
			}
		}
	});

	// Watch for recordId or flow changes
	$: if (browser && recordId && flow) {
		fetchRecordData(recordId, flow);
	}
</script>

<svelte:head>
	<title>Score: {recordId} | {flow} | Toxicity Analysis</title>
</svelte:head>

<div class="record-score-page">
	{#if loading}
		<div class="terminal-loading">
			<div class="loading-spinner">Loading record data...</div>
		</div>
	{:else if error}
		<div class="terminal-error">
			<h3>Error Loading Record</h3>
			<p>{error}</p>
		</div>
	{:else if recordData}
		<div class="record-header">
			<h1 class="record-title">
				<span class="record-id-badge">{recordData.id}</span>
				{recordData.name}
			</h1>
			<div class="flow-badge">{flow}</div>
		</div>

		<!-- Record Content Display -->
		<div class="section">
			<h2 class="section-title">Content Under Analysis</h2>
			<RecordDisplay {recordData} />
		</div>

		<!-- Toxicity Scores Summary -->
		<div class="section">
			<h2 class="section-title">Toxicity Score Summary</h2>
			<ToxicityScoreTable scores={recordData.toxicity_scores} />
		</div>

		<!-- Detailed AI Responses -->
		<div class="section">
			<h2 class="section-title">AI Model Responses</h2>
			<ScoreMessagesDisplay messages={recordData.messages} />
		</div>
	{:else}
		<div class="terminal-warning">
			<h3>Record Not Found</h3>
			<p>The requested record '{recordId}' could not be found in flow '{flow}'.</p>
		</div>
	{/if}
</div>
