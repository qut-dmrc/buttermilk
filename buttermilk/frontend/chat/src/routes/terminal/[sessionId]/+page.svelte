<script lang="ts">
  import { browser } from '$app/environment';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { onDestroy, onMount } from 'svelte';
  import ChatTerminal from '$lib/ChatTerminal.svelte';
  import {
    initializeApp,
    selectedCriteria,
    selectedFlow,
    selectedRecord
  } from '$lib/stores/apiStore';
  import { sessionId as sessionIdStore } from '$lib/stores/sessionStore';
  import { runFlowAction } from '$lib/stores/terminalActionsStore';
  
  // State variables for connection
  let isLoading = true;
  let error = '';
  let wsUrl = ''; // Direct WebSocket URL
  let isRestoringSession = false;
  let restorationComplete = false;
  
  // WebSocket terminal instance
  let websocketTerminal: ChatTerminal | null = null;
  
  // Get session ID from URL params
  $: urlSessionId = $page.params.sessionId;
  
  // Function to restore session messages
  async function restoreSessionMessages(sessionId: string) {
    isRestoringSession = true;
    try {
      const response = await fetch(`/api/session/${sessionId}/messages`);
      if (response.ok) {
        const data = await response.json();
        console.log(`Restoring ${data.messages.length} messages for session ${sessionId}`);
        
        // Messages will be restored via WebSocket after connection
        // Store them temporarily for restoration after WebSocket connects
        if (websocketTerminal && data.messages.length > 0) {
          // Send each message to the terminal for display
          for (const message of data.messages) {
            websocketTerminal.handleMessage(message);
          }
        }
        restorationComplete = true;
      } else if (response.status === 404) {
        console.log('Session not found, starting fresh');
        // Session doesn't exist yet, that's OK - it's a new session
        restorationComplete = true;
      }
    } catch (err) {
      console.error('Failed to restore session:', err);
      error = 'Failed to restore session history';
    } finally {
      isRestoringSession = false;
    }
  }
  
  // Function to run flow - uses global stores
  function runFlow() {
    // Read values directly from stores when function is called
    console.log('Running flow with:', { 
      flow: $selectedFlow, 
      record: $selectedRecord, 
      criteria: $selectedCriteria 
    });
    
    // Ensure flow is selected
    if (!$selectedFlow) {
      console.error('Cannot run flow: Flow not selected');
      return;
    }
    
    websocketTerminal?.sendRunFlowRequest($selectedFlow, $selectedRecord, $selectedCriteria);
  }
  
  onMount(async () => {
    // Update session store with URL session ID
    if (urlSessionId) {
      sessionIdStore.set(urlSessionId);
    }
    
    // Initialize the application
    await initializeApp(urlSessionId);
    
    // Build WebSocket URL with session ID from URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    wsUrl = `${protocol}//${host}/ws/${urlSessionId}`;
    
    console.log('WebSocket URL:', wsUrl);
    console.log('Session ID from URL:', urlSessionId);
    
    // Restore session messages if session exists
    await restoreSessionMessages(urlSessionId);
    
    isLoading = false;
    
    // Set up the action for running flows
    runFlowAction.set(runFlow);
  });
  
  onDestroy(() => {
    // Clear the action when component is destroyed
    runFlowAction.set(() => {});
  });
  
  // Handle terminal ready event
  function handleTerminalReady(event: CustomEvent) {
    websocketTerminal = event.detail;
    console.log('Terminal ready, session:', urlSessionId);
    
    // If we have pending restoration messages and terminal is now ready
    if (isRestoringSession || !restorationComplete) {
      // Terminal will handle restoration after connection
      restoreSessionMessages(urlSessionId);
    }
  }
</script>

<svelte:head>
  <title>Buttermilk Terminal - Session {urlSessionId}</title>
</svelte:head>

{#if isLoading}
  <div class="flex items-center justify-center h-full">
    <div class="text-center">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto mb-4"></div>
      <p class="text-gray-600">Initializing session...</p>
      {#if isRestoringSession}
        <p class="text-sm text-gray-500 mt-2">Restoring previous messages...</p>
      {/if}
    </div>
  </div>
{:else if error}
  <div class="flex items-center justify-center h-full">
    <div class="text-center text-red-600">
      <p class="font-semibold">Connection Error</p>
      <p class="text-sm mt-2">{error}</p>
      <button 
        class="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        on:click={() => window.location.reload()}
      >
        Retry
      </button>
    </div>
  </div>
{:else}
  <ChatTerminal 
    {wsUrl}
    bind:this={websocketTerminal}
    on:ready={handleTerminalReady}
  />
{/if}