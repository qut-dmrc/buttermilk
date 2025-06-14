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
        goto(`/score/${encodeURIComponent(currentFlow)}/${encodeURIComponent(currentDataset)}/${encodeURIComponent(recordId)}`, { replaceState: true });
      } else {
        // Redirect to /score/{flow}/{record_id}
        goto(`/score/${encodeURIComponent(currentFlow)}/${encodeURIComponent(recordId)}`, { replaceState: true });
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
      <p>Please access the score page via the proper navigation or include a 'flow' parameter in the URL.</p>
      <p>Expected format: <code>/score/&lt;flow&gt;/&lt;record_id&gt;</code> or <code>/score/&lt;flow&gt;/&lt;dataset&gt;/&lt;record_id&gt;</code></p>
      <div class="help-links">
        <a href="/score" class="help-link">Return to Score Home</a>
      </div>
    </div>
  {/if}
</div>

<style>
  .record-score-page {
    padding: 1rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    min-height: 80vh;
  }

  .record-header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(0, 255, 0, 0.3);
  }

  .record-title {
    color: #00ff00;
    font-size: 2rem;
    font-weight: bold;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .record-id-badge {
    background-color: rgba(0, 255, 255, 0.2);
    color: #00ffff;
    padding: 0.25rem 0.5rem;
    border: 1px solid #00ffff;
    font-size: 0.8em;
    border-radius: 3px;
  }

  .section {
    margin-bottom: 3rem;
  }

  .section-title {
    color: #00ffff;
    font-size: 1.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    text-transform: uppercase;
  }

  .terminal-loading,
  .terminal-error,
  .terminal-warning {
    text-align: center;
    padding: 3rem;
    background-color: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    margin: 2rem 0;
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

  .loading-spinner {
    font-size: 1.2rem;
  }

  .terminal-error h3,
  .terminal-warning h3 {
    margin-bottom: 1rem;
    font-size: 1.5rem;
  }

  .terminal-error p,
  .terminal-warning p {
    font-size: 1rem;
    opacity: 0.8;
  }

  .terminal-loading p {
    font-size: 1rem;
    opacity: 0.8;
    margin-top: 1rem;
  }

  .help-links {
    margin-top: 1.5rem;
  }

  .help-link {
    display: inline-block;
    padding: 0.5rem 1rem;
    background-color: rgba(0, 255, 0, 0.2);
    color: #00ff00;
    text-decoration: none;
    border: 1px solid #00ff00;
    border-radius: 3px;
    font-weight: bold;
    transition: all 0.2s ease;
  }

  .help-link:hover {
    background-color: rgba(0, 255, 0, 0.3);
    color: #ffffff;
  }

  code {
    background-color: rgba(255, 255, 255, 0.1);
    color: #00ffff;
    padding: 0.2rem 0.4rem;
    border-radius: 2px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }
</style>