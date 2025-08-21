<script lang="ts">
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { selectedFlow, selectedDataset, initializeApp } from '$lib/stores/apiStore';

	let recordId: string;
	let currentFlow: string = '';
	let currentDataset: string = '';
	let redirecting = true;

	$: recordId = $page.params.record_id;
	$: currentFlow = $selectedFlow || $page.url.searchParams.get('flow') || '';
	$: currentDataset = $selectedDataset || $page.url.searchParams.get('dataset') || '';

	// Redirect to new URL pattern
	async function redirectToNewPattern() {
		if (!recordId) return;

		// If no flow is specified, try to get from store or use default
		if (!currentFlow) {
			// Wait for app initialization to get available flows
			await initializeApp();
			// Use selectedFlow from store if available, otherwise redirect will show error
			currentFlow = $selectedFlow || '';
		}

		if (currentFlow) {
			if (currentDataset) {
				// Redirect to /score/{flow}/{dataset}/{record_id}
				goto(
					`/score/${encodeURIComponent(currentFlow)}/${encodeURIComponent(currentDataset)}/${encodeURIComponent(recordId)}`,
					{ replaceState: true }
				);
			} else {
				// Redirect to /score/{flow}/{record_id}
				goto(`/score/${encodeURIComponent(currentFlow)}/${encodeURIComponent(recordId)}`, {
					replaceState: true
				});
			}
		} else {
			// If no flow available, show error instead of redirecting
			redirecting = false;
		}
	}

	onMount(() => {
		if (browser) {
			redirectToNewPattern();
		}
	});

	// Watch for parameter changes to redirect
	$: if (browser && recordId) {
		redirectToNewPattern();
	}
</script>

<svelte:head>
	<title>Score: {recordId} | Redirecting...</title>
</svelte:head>

<div class="record-score-page">
	{#if redirecting}
		<div class="terminal-loading">
			<div class="loading-spinner">Redirecting to new URL structure...</div>
			<p>You are being redirected to the new score page format.</p>
		</div>
	{:else}
		<div class="terminal-error">
			<h3>Cannot Redirect - Flow Required</h3>
			<p>The score page now requires a flow parameter in the URL.</p>
			<p>
				Please access the score page via the proper navigation or include a 'flow' parameter in the
				URL.
			</p>
			<p>
				Expected format: <code>/score/&lt;flow&gt;/&lt;record_id&gt;</code> or
				<code>/score/&lt;flow&gt;/&lt;dataset&gt;/&lt;record_id&gt;</code>
			</p>
			<div class="help-links">
				<a href="/score" class="help-link">Return to Score Home</a>
			</div>
		</div>
	{/if}
</div>
