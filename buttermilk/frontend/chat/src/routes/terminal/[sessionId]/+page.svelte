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
  
  // Store pending restoration messages if terminal isn't ready yet
  let pendingMessages: any[] = [];
  
  // Function to restore session messages
  async function restoreSessionMessages(sessionId: string) {
    isRestoringSession = true;
    try {
      const response = await fetch(`/api/session/${sessionId}/messages`);
      if (response.ok) {
        const data = await response.json();
        // The API returns the messages array directly, not wrapped in a messages property
        const messages = Array.isArray(data) ? data : (data.messages || []);
        console.log(`Restoring ${messages.length} messages for session ${sessionId}`);
        
        if (messages.length > 0) {
          if (websocketTerminal) {
            // Terminal is ready, restore messages immediately
            for (const message of messages) {
              websocketTerminal.handleMessage(message);
            }
          } else {
            // Terminal not ready yet, store for later
            pendingMessages = messages;
            console.log(`Stored ${pendingMessages.length} pending messages for restoration`);
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
    const terminal = event.detail;
    // Store the full terminal reference, not just handleMessage
    websocketTerminal = terminal;
    console.log('Terminal ready, session:', urlSessionId);
    
    // Process any pending messages that were fetched before terminal was ready
    if (pendingMessages.length > 0) {
      console.log(`Processing ${pendingMessages.length} pending messages`);
      for (const message of pendingMessages) {
        terminal.handleMessage(message);
      }
      pendingMessages = []; // Clear pending messages
    }
    
    // If restoration is still in progress or not complete, trigger it again
    if (isRestoringSession || (!restorationComplete && pendingMessages.length === 0)) {
      console.log('Terminal ready but restoration not complete, re-triggering restoration');
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