<script lang="ts">
	import { browser } from '$app/environment';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { onDestroy, onMount } from 'svelte';
	import ChatTerminal from '$lib/ChatTerminal.svelte';
	import {
		initializeApp,
		selectedCriteria,
		selectedDataset,
		selectedFlow,
		selectedRecord,
		isDemoMode
	} from '$lib/stores/apiStore';
	import { sessionId as sessionIdStore } from '$lib/stores/sessionStore';
	import { runFlowAction } from '$lib/stores/terminalActionsStore';

	// State variables for connection
	let isLoading = true;
	let isDemoModeInitialized = false; // Track if demo mode detection is complete
	let error = '';
	let wsUrl = ''; // Direct WebSocket URL
	let isRestoringSession = false;
	let restorationComplete = false;
	let messagesProcessed = false; // Track if messages have been processed to prevent duplicates
	let sessionMetadata: any = null;
	let lastInitializedSessionId = ''; // Track last initialized session to avoid re-initialization
	let isWebSocketReady = false; // Track WebSocket connection status
	
	// Message duplicate tracking
	let processedMessageIds = new Set<string>();

	// WebSocket terminal instance
	let websocketTerminal: ChatTerminal | null = null;
	
	// Helper function to filter out duplicate messages
	function filterDuplicateMessages(messages: any[]): any[] {
		const uniqueMessages = [];
		for (const message of messages) {
			if (message.message_id && !processedMessageIds.has(message.message_id)) {
				uniqueMessages.push(message);
				processedMessageIds.add(message.message_id);
			} else if (!message.message_id) {
				// If message doesn't have an ID, add it but log a warning
				console.warn('Message without ID detected:', message);
				uniqueMessages.push(message);
			}
		}
		
		if (uniqueMessages.length !== messages.length) {
			console.log(`Filtered out ${messages.length - uniqueMessages.length} duplicate messages`);
		}
		
		return uniqueMessages;
	}

	// Get session ID from URL params
	$: urlSessionId = $page.params.sessionId;

	// React to session ID changes and fully re-initialize session
	$: if (urlSessionId && browser && urlSessionId !== lastInitializedSessionId) {
		console.log('Session ID changed to:', urlSessionId);
		// Fully initialize the new session
		initializeSession(urlSessionId);
	}

	// Demo mode: Log parameter selection status (manual reload via button now)
	$: {
		if ($isDemoMode && browser) {
			console.log('Demo mode parameter selection:', {
				flow: $selectedFlow || 'not selected',
				dataset: $selectedDataset || 'not selected', 
				record: $selectedRecord || 'not selected',
				criteria: $selectedCriteria || 'not selected',
				allSelected: !!(($selectedFlow && $selectedDataset && $selectedRecord && $selectedCriteria))
			});
		}
	}


	// Store pending restoration messages if terminal isn't ready yet
	let pendingMessages: any[] = [];

	// Function to initialize a session (used by onMount and reactive statements)
	async function initializeSession(sessionId: string) {
		// Track this session as initialized to avoid re-initialization
		lastInitializedSessionId = sessionId;
		
		// Reset state
		isLoading = true;
		error = '';
		sessionMetadata = null;
		isDemoModeInitialized = false;
		messagesProcessed = false;
		restorationComplete = false;
		isWebSocketReady = false;
		
		// Reset message duplicate tracking for new session
		processedMessageIds.clear();
		pendingMessages = [];
		
		// Update session store
		sessionIdStore.set(sessionId);

		// Initialize the application and wait for demo mode detection to complete
		await initializeApp(sessionId);
		
		// Demo mode detection is now complete
		isDemoModeInitialized = true;

		// Build WebSocket URL with session ID
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const host = window.location.host;
		wsUrl = `${protocol}//${host}/ws/${sessionId}`;

		console.log('WebSocket URL:', wsUrl);
		console.log('Session ID from URL:', sessionId);

		// Restore session messages if session exists
		await restoreSessionMessages(sessionId);

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
	}

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

				console.log(
					`Session ${sessionId} status: ${sessionMetadata.flow_status}, resumable: ${sessionMetadata.is_resumable}`
				);

				// Always load messages for viewing, but store metadata for UI decisions

				console.log(`Restoring ${messages.length} messages for session ${sessionId}`);

				if (messages.length > 0 && !messagesProcessed) {
					// Filter out any duplicate messages before storing
					const filteredMessages = filterDuplicateMessages(messages);
					
					if (filteredMessages.length > 0) {
						if (websocketTerminal) {
							// Terminal component exists, but in demo/readonly mode we need to wait for it to be fully ready
							// Store as pending messages to ensure proper rendering pipeline
							console.log('Terminal exists but storing as pending messages to ensure proper rendering');
							pendingMessages = filteredMessages;
							console.log(`Stored ${pendingMessages.length} pending messages for restoration`);
						} else {
							// Terminal not ready yet, store for later (don't set messagesProcessed yet)
							console.log('Terminal not ready, storing as pending messages');
							pendingMessages = filteredMessages;
							console.log(`Stored ${pendingMessages.length} pending messages for restoration`);
						}
					} else {
						console.log('All messages were duplicates, no new messages to process');
					}
				} else if (messagesProcessed) {
					console.log('Messages already processed, skipping duplicate restoration');
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
		// Disable running flows in demo mode
		if ($isDemoMode) {
			console.log('Cannot run flow: Demo mode active - backend not available');
			return;
		}

		// Read values directly from stores when function is called
		console.log('Running flow with:', {
			flow: $selectedFlow,
			dataset: $selectedDataset,
			record: $selectedRecord,
			criteria: $selectedCriteria
		});

		// Ensure flow is selected
		if (!$selectedFlow) {
			console.error('Cannot run flow: Flow not selected');
			return;
		}

		websocketTerminal?.sendRunFlowRequest($selectedFlow, $selectedDataset, $selectedRecord, $selectedCriteria);
	}

	onMount(async () => {
		// Initialize the session
		if (urlSessionId) {
			await initializeSession(urlSessionId);
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
		isWebSocketReady = true; // Mark WebSocket as ready
		console.log('Terminal ready, session:', urlSessionId);

		// Process any pending messages that were fetched before terminal was ready
		if (pendingMessages.length > 0 && !messagesProcessed) {
			console.log(`Processing ${pendingMessages.length} pending messages`);
			for (const message of pendingMessages) {
				terminal.handleMessage(message);
			}
			messagesProcessed = true;
			pendingMessages = []; // Clear pending messages
		} else if (messagesProcessed) {
			console.log('Messages already processed, skipping pending message processing');
		}

		// Only trigger restoration if it hasn't been attempted yet and messages haven't been processed
		if (isRestoringSession) {
			console.log('Terminal ready but restoration still in progress, waiting...');
		} else if (!restorationComplete && !messagesProcessed && pendingMessages.length === 0) {
			console.log('Terminal ready but restoration not complete, re-triggering restoration');
			restoreSessionMessages(urlSessionId);
		} else if (messagesProcessed) {
			console.log('Terminal ready and messages already processed, no action needed');
		}
	}
</script>

<svelte:head>
	<title>Buttermilk Terminal - Session {urlSessionId}</title>
</svelte:head>

{#if isLoading || !isDemoModeInitialized}
	<div class="flex items-center justify-center h-full">
		<div class="text-center">
			<div
				class="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto mb-4"
			></div>
			<p class="text-gray-600">Initializing session...</p>
			{#if !isDemoModeInitialized}
				<p class="text-sm text-gray-500 mt-2">Detecting mode...</p>
			{:else if isRestoringSession}
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
		readonly={$isDemoMode || (sessionMetadata && !sessionMetadata.is_resumable)}
		currentSessionId={urlSessionId}
		sessionStatus={$isDemoMode ? 'demo' : (isWebSocketReady ? 'ready' : (sessionMetadata?.flow_status || 'unknown'))}
		isResumable={!$isDemoMode && sessionMetadata?.is_resumable !== false}
		bind:this={websocketTerminal}
		on:ready={handleTerminalReady}
	/>
{/if}
