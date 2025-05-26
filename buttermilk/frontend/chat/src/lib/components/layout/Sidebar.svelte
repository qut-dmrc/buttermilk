<script lang="ts">
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import ApiDropdown from '$lib/ApiDropdown.svelte';
	import {
		flowChoices,
		recordsStore,
		criteriaStore,
		modelStore,
		selectedFlow,
		selectedRecord,
		selectedCriteria,
		selectedModel,
		initializeApp
	} from '$lib/stores/apiStore';
	import { runFlowAction } from '$lib/stores/terminalActionsStore';
	
	$: isTerminalPage = $page.route.id === '/terminal';

    import { flowRunning } from '$lib/stores/apiStore';
	// Initialize app data when we're on the terminal page and the sidebar mounts
	onMount(() => {
		if (isTerminalPage) {
			initializeApp();
		}
	});

  function runFlow() {
    flowRunning.set(true); // Update the store
    $runFlowAction && $runFlowAction();
  }
</script>

<!-- Sidebar component -->
{#if isTerminalPage}
	<!-- Terminal API Selector Section -->
	<div class="terminal-selector p-3 mb-4">
		<h2 class="mb-3">API Selector</h2>
			<div>
				<!-- Flows Dropdown -->
				<ApiDropdown
					store={flowChoices}
					label="Select Flow"
					placeholder="Choose a flow..."
					bind:value={$selectedFlow}
					isPlainArray={true}
				/>

				<!-- Records Dropdown - only enabled if a flow is selected -->
				<ApiDropdown
					store={recordsStore}
					label="Select Record"
					placeholder="Choose a record..."
					bind:value={$selectedRecord}
					disabled={!$selectedFlow}
					valueProperty="record_id"
					labelProperty="name"
					on:change={(e) => {
						console.log("Record dropdown changed detail:", e.detail);
						console.log("Record dropdown changed target value:", e.target ? (e.target as HTMLSelectElement).value : "No target");
						console.log("Bound selectedRecord after change:", $selectedRecord);
					}}
				/>
				<!-- Criteria Dropdown - only enabled if a flow is selected -->
				<ApiDropdown
					store={criteriaStore}
					label="Select Criteria"
					placeholder="Choose criteria..."
					bind:value={$selectedCriteria}
					disabled={!$selectedFlow}
					isPlainArray={true}
				/>
				<!-- Models Dropdown - only enabled if a flow is selected -->
				<ApiDropdown
					store={modelStore}
					label="Select Model(s)"
					placeholder="Choose model(s)..."
					bind:value={$selectedModel}
					disabled={!$selectedFlow}
					isPlainArray={true}
				/>
				<!-- Add standard Bootstrap button classes -->
						<!-- disabled={!$selectedFlow || !$selectedRecord} -->
				<div class="mt-3">
					<button
						class="btn btn-primary w-100"
						onclick={runFlow}
					>
						Run flow
					</button>
				</div>
			</div>
	</div>
{:else}
	<!-- Default Sidebar Content -->
	<div class="alert alert-secondary mt-5 col-sm-7" role="alert">
		Warning: explicit content and hateful ideologies here!
	</div>

	<!-- Placeholder for dynamic content from the original sidebar -->
	<div>
		<h5 class="header">Drag Queens vs White Supremacists examples:</h5>
		<ul>
			<li><a href="/scores/placeholder-drag-1">Example Drag Link 1</a></li>
			<li><a href="/scores/placeholder-drag-2">Example Drag Link 2</a></li>
			<!-- Add more placeholders if needed -->
		</ul>
	</div>

	<div>
		<h5 class="header">Oversight Board examples:</h5>
		<ul>
			<li><a href="/scores/placeholder-osb-1">Example OSB Link 1</a></li>
			<li><a href="/scores/placeholder-osb-2">Example OSB Link 2</a></li>
			<!-- Add more placeholders if needed -->
		</ul>
	</div>

	<div>
		<h5 class="header">Tone Policing examples:</h5>
		<ul>
			<li><a href="/scores/placeholder-tone-1">Example Tone Link 1</a></li>
			<li><a href="/scores/placeholder-tone-2">Example Tone Link 2</a></li>
			<!-- Add more placeholders if needed -->
		</ul>
	</div>
{/if}