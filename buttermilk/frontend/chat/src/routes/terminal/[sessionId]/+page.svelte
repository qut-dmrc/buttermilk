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
  let sessionMetadata = null;
  
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
        
        // Extract messages and metadata from the new API response format
        const messages = data.messages || [];
        sessionMetadata = data.session_metadata || {};
        
        console.log(`Session ${sessionId} status: ${sessionMetadata.flow_status}, resumable: ${sessionMetadata.is_resumable}`);
        
        // Always load messages for viewing, but store metadata for UI decisions
        
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
    
    // Set up the action for running flows (only if session is resumable)
    if (!sessionMetadata || sessionMetadata.is_resumable) {
      runFlowAction.set(runFlow);
    } else {
      // Disable run flow action for completed sessions
      runFlowAction.set(() => {
        console.log('Cannot run flow: Session is completed');
      });
    }
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

<!-- Session Status Indicator -->
{#if sessionMetadata && !isLoading}
  <div class="session-status-bar">
    <div class="session-info">
      <span class="session-id">Session: {urlSessionId.slice(0, 8)}...</span>
      <div class="status-indicator status-{sessionMetadata.flow_status}">
        {#if sessionMetadata.flow_status === 'running'}
          🟢 Active
        {:else if sessionMetadata.flow_status === 'completed'}
          ✓ Completed
        {:else if sessionMetadata.flow_status === 'failed'}
          ❌ Failed
        {:else}
          ⚫ {sessionMetadata.flow_status}
        {/if}
      </div>
      {#if !sessionMetadata.is_resumable}
        <span class="readonly-badge">Read-only</span>
      {/if}
    </div>
    {#if !sessionMetadata.is_resumable}
      <button 
        class="new-session-btn"
        on:click={() => window.location.href = '/terminal'}
      >
        Start New Session
      </button>
    {/if}
  </div>
{/if}

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
    selectedFlow={$selectedFlow || ''}
    selectedRecord={$selectedRecord || ''}
    readonly={sessionMetadata && !sessionMetadata.is_resumable}
    bind:this={websocketTerminal}
    on:ready={handleTerminalReady}
  />
{/if}

<style>
  .session-status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 1rem;
    background: rgba(0, 0, 0, 0.8);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 0.85rem;
  }

  .session-info {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .session-id {
    color: #888;
    font-size: 0.8rem;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    font-weight: bold;
    font-size: 0.85rem;
  }

  .status-running {
    color: #00ff00;
  }

  .status-completed {
    color: #00ffff;
  }

  .status-failed {
    color: #ff4444;
  }

  .readonly-badge {
    background: rgba(255, 170, 0, 0.2);
    color: #ffaa00;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    border: 1px solid rgba(255, 170, 0, 0.3);
  }

  .new-session-btn {
    background: rgba(0, 255, 0, 0.1);
    border: 1px solid #00ff00;
    color: #00ff00;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .new-session-btn:hover {
    background: rgba(0, 255, 0, 0.2);
    box-shadow: 0 0 5px rgba(0, 255, 0, 0.3);
  }
</style>