<script lang="ts">
  import { browser } from '$app/environment';
import { onMount, onDestroy } from 'svelte'; // Import onDestroy
import DirectWebSocketTerminal from '$lib/DirectWebSocketTerminal.svelte';
import {
  // Import the necessary stores
  selectedFlow,
  selectedRecord, // Import global selection store
  selectedCriteria, // Import global selection store
  selectedModel, // Import global selection store
  initializeApp // Import initializer
} from '$lib/stores/apiStore';
import { runFlowAction } from '$lib/stores/terminalActionsStore'; // Import the action store

// State variables for connection
let isLoading = true;
let error = '';
let sessionId = '';
let wsUrl = ''; // Direct WebSocket URL

// WebSocket terminal instance
let websocketTerminal: DirectWebSocketTerminal | null = null; // Initialize as null

// Function to run flow - uses global stores now
function runFlow() {
  // Read values directly from stores when function is called
  console.log('Running flow with:', { 
    flow: $selectedFlow, 
    record: $selectedRecord, 
    criteria: $selectedCriteria 
  });
  
  // Extra debug logging
  console.log('Record value details:', {
    value: $selectedRecord,
    type: typeof $selectedRecord,
    length: $selectedRecord?.length || 0,
    isEmpty: !$selectedRecord
  });
  
  // Ensure record is selected and has a value
  if (!$selectedFlow) {
    console.error('Cannot run flow: Flow not selected');
    return;
  }
  
  websocketTerminal?.sendRunFlowRequest($selectedFlow, $selectedRecord, $selectedCriteria);
}

// Connect to the backend with a session ID
onMount(async () => {
  if (browser) {
    // Only import and use Bootstrap JS on the client
    import('bootstrap/dist/js/bootstrap.bundle.min.js').then(bootstrap => {
      // Initialize components if necessary
      console.log('Bootstrap JS loaded on client');
    });
  }

  // Fetch the flow config data first (uses cache if valid)
  initializeApp();

  // Then proceed with session setup
  try {
    console.log('Fetching session ID from backend');

    // Fetch a session ID directly from the backend
    const response = await fetch('/api/session');

    if (!response.ok) {
      throw new Error(`Failed to get session: ${response.statusText}`);
    }

    const sessionData = await response.json() as { session_id?: string };

    console.log('Session data:', sessionData);

    if (!sessionData.session_id) {
      throw new Error('No session ID returned from backend');
    }

    // Store the session ID
    sessionId = sessionData.session_id;
    console.log('Using session ID:', sessionId);

    // Construct the WebSocket URL - use the proxy defined in vite.config.ts
    // This allows Vite to handle the WebSocket connection properly
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // Includes hostname + port if any
    wsUrl = `${protocol}//${host}/ws`; // Pass the base URL without session ID
    console.log('WebSocket URL set to:', wsUrl);

// Set the runFlow function to the store regardless of WebSocket connection status
// This allows the "Run flow" button to become enabled once selections are made
runFlowAction.set(runFlow);

  } catch (err) {
    console.error('Error setting up connection:', err);
    error = err instanceof Error ? err.message : 'Unknown error';
  } finally {
    isLoading = false;
  }
});

// Clean up the action store when the component is destroyed
onDestroy(() => {
  runFlowAction.set(null);
});
</script>

<!-- Just the terminal component, sidebar is in layout -->
<div class="h-100">
  {#if wsUrl}
  <DirectWebSocketTerminal
    wsUrl={wsUrl}
    bind:this={websocketTerminal}
    selectedFlow={$selectedFlow}
    selectedRecord={$selectedRecord} 
  />
  {:else}
    <div>Loading WebSocket configuration...</div>
  {/if}
</div>
