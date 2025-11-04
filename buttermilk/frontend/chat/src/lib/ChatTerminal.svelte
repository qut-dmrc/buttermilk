<script lang="ts">
	import { onDestroy, onMount, createEventDispatcher } from 'svelte';
	import { get } from 'svelte/store';
	import MessageDisplay from './components/MessageDisplay.svelte';
	import {
		flowRunning,
		selectedFlow as flowStore,
		selectedRecord as recordStore,
		selectedCriteria
	} from './stores/apiStore';
	import { messageHistory } from './stores/messageHistoryStore';
	import { addMessage as addToMessageStore } from './stores/messageStore';
	import { sessionId } from './stores/sessionStore';
	import { tokenUsageDisplay } from './stores/tokenUsageStore';
	import './styles/terminal.scss';
	import {
		type SystemPromptMessage,
		type UserResponseMessage,
		type Message,
		type MessageType,
		type SystemUpdate,
		createUserResponse,
		isSystemUpdate,
		normalizeWebSocketMessage
	} from './utils/messageUtils';
	import { checkBackendHealth, logBackendStatus } from './utils/backendUtils';

	// Event dispatcher for communicating with parent
	const dispatch = createEventDispatcher();

	// WebSocket connection parameters
	export let wsUrl: string;
	export let selectedFlow: string; // Prop for selected flow
	export let selectedRecord: string; // Prop for selected record
	export let readonly: boolean = false; // Readonly mode for completed sessions
	export let currentSessionId: string = ''; // Session ID from URL
	export let sessionStatus: string = 'unknown'; // Session status
	export let isResumable: boolean = true; // Whether session is resumable

	// Monitor props for debugging
	$: {
		console.debug(
			`ChatTerminal props updated: selectedFlow='${selectedFlow}', selectedRecord='${selectedRecord}'`
		);
	}

	// Compute unified status that combines connection and flow status
	$: displayStatus = !isConnected && !readonly && sessionStatus !== 'demo'
		? (connectionError?.includes('terminated') ? 'terminated' :
		   isReconnecting ? 'reconnecting' : 'disconnected')
		: (sessionStatus === 'unknown' ? 'idle' : sessionStatus);

	// Component state
	let socket: WebSocket | null = null;
	let messages: Message[] = [];
	let displayedMessageIds = new Set<string>(); // Track message IDs to prevent duplicates
	let inputMessage = '';
	let isConnected = false;
	let connectionError = '';
	let isInterruptEnabled = false;
	let humanInLoop = true;
	let messageListElement: HTMLDivElement; // To control scrolling
	export let selectedTheme = 'theme-term'; // Default theme
	let isReconnecting = false; // Track reconnection attempts
	let reconnectAttempts = 0; // Count reconnection attempts
	let reconnectTimeout: number | null = null; // Track reconnection timeout
	const MAX_RECONNECT_ATTEMPTS = 10; // Maximum reconnection attempts

	// System update state
	let systemUpdateStatus: SystemUpdate | null = null;

	// System prompt state
	let currentUIMessage: SystemPromptMessage | null = null;
	let selectionOptions: string[] = [];
	let isConfirmRequest = false;

	// Load stored messages for the current session
	function loadStoredMessages(sessionIdToLoad: string) {
		try {
			const key = `messageHistory_${sessionIdToLoad}`;
			const stored = localStorage.getItem(key);
			if (stored) {
				const storedMessages: Message[] = JSON.parse(stored);
				console.debug(
					`Loading ${storedMessages.length} stored messages for session ${sessionIdToLoad}`
				);

				// Set the messages array directly to avoid any reprocessing
				messages = storedMessages;

				// Restore system prompt state from the last system_prompt if any
				const lastUIMessage = storedMessages
					.slice()
					.reverse()
					.find((msg) => msg.type === 'system_prompt');
				if (lastUIMessage && lastUIMessage.outputs) {
					currentUIMessage = lastUIMessage.outputs as SystemPromptMessage;

					// Determine input type
					const inputs = currentUIMessage.options;

					// Check if it's a selection request
					if (inputs && Array.isArray(inputs) && inputs.length > 0) {
						selectionOptions = inputs.map((val) => String(val));
						isConfirmRequest = false;
					} else {
						selectionOptions = [];
						isConfirmRequest = true;
					}

					console.debug('Restored manager request state from stored messages:', {
						selectionOptions,
						isConfirmRequest
					});
				}

				// Scroll to bottom after messages are loaded
				setTimeout(() => {
					if (messageListElement) {
						messageListElement.scrollTop = messageListElement.scrollHeight;
					}
				}, 100);

				console.debug(`Loaded ${storedMessages.length} previous messages`);
			} else {
				console.debug(`No stored messages found for session ${sessionIdToLoad}`);
			}
		} catch (e) {
			console.error('Error loading stored messages:', e);
			// Don't call addSystemMessage here to avoid any reprocessing
		}
	}

	// Handle component mounting
	onMount(async () => {
		// Make the callback async
		console.debug('Attempting direct WebSocket connection to:', wsUrl);

		// If a session ID was provided via props (from URL), use that
		if (currentSessionId) {
			console.debug('Using session ID from URL prop:', currentSessionId);
			sessionId.set(currentSessionId);
		} else if (!get(sessionId)) {
			// Only create a new session if no session ID exists at all
			addSystemMessage('Initializing session...');
			await getNewSessionId(); // Fetches and stores a clean session ID
		}
		if (get(sessionId)) {
			// Only load localStorage messages if this isn't a session-based URL
			// (session pages handle restoration via backend API)
			const currentUrl = window.location.pathname;
			if (!currentUrl.includes('/terminal/')) {
				// Load stored messages for this session from localStorage
				loadStoredMessages(get(sessionId));
			}

			// Only connect WebSocket if not in readonly mode and not demo mode
			if (!readonly && sessionStatus !== 'demo') {
				// wsUrl prop should be the base like "ws://localhost:5173/ws"
				console.debug('Attempting direct WebSocket connection. Base wsUrl prop:', wsUrl);
				connectWebSocket();
			} else {
				console.debug('Terminal in readonly mode or demo mode, skipping WebSocket connection');
				// Emit ready event even in readonly mode so parent can process messages
				setTimeout(() => {
					dispatch('ready', { handleMessage, sendRunFlowRequest });
				}, 100);
			}
		} else {
			const errorMsg =
				'Failed to obtain session ID on mount. WebSocket connection not established.';
			connectionError = errorMsg;
			addSystemMessage(`Error: ${errorMsg}`);
			console.error(errorMsg);
		}

		document.body.className = selectedTheme; // Apply default theme to body
	});

	// Handle component destruction
	onDestroy(() => {
		// Clear any pending reconnection timeout
		if (reconnectTimeout) {
			clearTimeout(reconnectTimeout);
			reconnectTimeout = null;
		}

		if (socket) {
			console.log('Closing WebSocket connection');
			socket.close();
		}
	});

	// Start a new session
	async function startNewSession() {
		// First close the existing connection
		if (socket) {
			try {
				socket.close();
			} catch (e) {
				console.error('Error closing socket:', e);
			}
		}

		// Clear both session ID and message history
		sessionId.clear();
		messageHistory.clearHistory();

		// Reset API selector state to bring back sidebar
		flowRunning.set(false);
		flowStore.set('');
		recordStore.set('');
		selectedCriteria.set('');

		// Clear messages and message ID tracking
		messages = [];
		displayedMessageIds.clear();
		// Get new session ID
		addSystemMessage('Starting a new session...');
		await getNewSessionId();
		// Reconnect
		connectWebSocket();
	}

	// Toggle auto-approve state
	function toggleAutoApprove() {
		humanInLoop = !humanInLoop;
		console.log('Toggled human in loop state:', humanInLoop, ' approving:', !humanInLoop);
		const message = createUserResponse(!humanInLoop, null, null, null, null, humanInLoop);
		sendUserResponse(message);
	}

	// Toggle interrupt state
	function toggleInterrupt() {
		isInterruptEnabled = !isInterruptEnabled;

		let message;
		if (isInterruptEnabled) {
			message = {
				type: 'TaskProcessingStarted',
				role: 'MANAGER',
				agent_id: 'web socket'
			};
			// Also send separate interrupt message
			const response = createUserResponse(false, null, null, null, true, humanInLoop);
			sendUserResponse(response);
		} else {
			message = {
				type: 'TaskProcessingComplete',
				role: 'MANAGER',
				agent_id: 'web socket'
			};
			const response = createUserResponse(false, null, null, null, false, humanInLoop);
			sendUserResponse(response);
		}

		try {
			console.log('Sending interrupt message:', message);
			if (socket && socket.readyState === WebSocket.OPEN) {
				socket.send(JSON.stringify(message));
			}
		} catch {
			console.error('Error sending interrupt start message:');
		}
	}

	// Function to send UserResponse back via WebSocket
	function handleUserResponse(event: CustomEvent<UserResponseMessage>) {
		const response = event.detail;
		console.debug('Received user response from component:', response);
		sendUserResponse(response);
	}

	function sendUserResponse(response: UserResponseMessage) {
		if (socket && socket.readyState === WebSocket.OPEN) {
			try {
				socket.send(JSON.stringify(response));
				console.log('Sent user response via WebSocket:', response);
				// Clear current system prompt after responding
				currentUIMessage = null;
				selectionOptions = [];
				isConfirmRequest = false;
			} catch (e) {
				console.error('Error sending user response:', e);
				addSystemMessage(`Error sending response: ${e}`);
			}
		} else {
			console.error('WebSocket not open, cannot send user response.');
			addSystemMessage('Error: Connection not open.');
		}
	}

	// Handle selection from options
	function handleSelection(value: string) {
		const response = createUserResponse(true, value, null, null, false, humanInLoop);
		sendUserResponse(response);
	}

	// Handle confirm/reject
	function handleConfirm(value: boolean) {
		const response = createUserResponse(value, null, null, null, false, humanInLoop);
		sendUserResponse(response);
	}

	// Handle halt
	function handleHalt() {
		const response = createUserResponse(false, null, null, true, null, humanInLoop);
		sendUserResponse(response);
	}
	async function getNewSessionId() {
		try {
			// Add a system message about new session
			addSystemMessage('Starting a new session...');

			let fetchedSessionId;

			// Define interface for session response
			interface SessionResponse {
				session_id: string;
				new?: boolean;
				error?: string;
			}

			const response = await fetch('/api/session');
			if (!response.ok) {
				throw new Error(`Failed to fetch session ID: ${response.statusText}`);
			}
			const data = (await response.json()) as SessionResponse;
			fetchedSessionId = data.session_id;

			// Store the session ID for future use
			if (fetchedSessionId) {
				sessionId.set(fetchedSessionId);
			} else {
				console.error('Fetched session ID is null or empty:', data);
				addSystemMessage('Error: Could not retrieve a valid session ID.');
			}
		} catch (error) {
			console.error('Error fetching new session ID:', error);
			addSystemMessage(`Error initializing session: ${error}`);
		}
	}

	// Connect to WebSocket with retry mechanism and fallback
	async function connectWebSocket() {
		// Clear any existing reconnection timeout
		if (reconnectTimeout) {
			clearTimeout(reconnectTimeout);
			reconnectTimeout = null;
		}

		// Check max attempts
		if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
			console.error(`Max reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) exceeded`);
			connectionError = `Failed to connect after ${MAX_RECONNECT_ATTEMPTS} attempts`;
			isReconnecting = false;
			return;
		}

		// Close any existing socket first
		if (socket) {
			try {
				socket.close();
			} catch (e) {
				console.error('Error closing existing socket:', e);
			}
		}

		// Get stored session ID if available
		const currentSessionId = get(sessionId);
		if (!currentSessionId) {
			connectionError = 'No session ID available to connect WebSocket.';
			addSystemMessage('Error: Cannot connect without a session ID.');
			console.error(connectionError);
			isReconnecting = false; // Stop reconnection attempts if no session ID
			return;
		}

		// Check if backend is available before attempting WebSocket connection
		const isBackendHealthy = await checkBackendHealth(false); // Don't use cache for WebSocket connections
		logBackendStatus('WebSocket connection', isBackendHealthy);

		if (!isBackendHealthy) {
			connectionError = 'Backend service unavailable';
			isReconnecting = false; // Stop reconnection attempts when backend is down

			// Schedule a health check retry with exponential backoff instead of WebSocket retry
			const backoffDelay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
			console.debug(`Will retry backend health check in ${backoffDelay}ms`);
			reconnectTimeout = setTimeout(() => {
				if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
					reconnectAttempts++;
					connectWebSocket();
				}
			}, backoffDelay) as unknown as number;
			return;
		}

		try {
			// wsUrl already includes the session ID from the parent page
			console.debug('Attempting to connect to WebSocket with URL:', wsUrl);

			// Set connection timeout
			const connectionTimeout = setTimeout(() => {
				if (!isConnected && socket) {
					console.warn('WebSocket connection timeout...');
					connectionError = 'Connection timeout...';

					// Force close the socket that's hanging
					try {
						socket.close();
					} catch (e) {
						console.error('Error closing timed out socket:', e);
					}
				}
			}, 5000);

			socket = new WebSocket(wsUrl);

			socket.onopen = () => {
				console.log('Direct WebSocket connection established');
				clearTimeout(connectionTimeout);
				isConnected = true;
				connectionError = '';
				isReconnecting = false; // Reset reconnection state
				reconnectAttempts = 0; // Reset reconnection counter
				// Connection status now appears in UI header instead of as system message

				// Emit ready event so parent can process pending messages
				dispatch('ready', { handleMessage, sendRunFlowRequest });
			};

			socket.onmessage = (event) => {
				try {
					// First try to parse the message as JSON
					let messageData: unknown;

					if (typeof event.data === 'string') {
						try {
							messageData = JSON.parse(event.data);
						} catch (parseError) {
							// Not valid JSON
							console.error('Message is not valid JSON, ignoring:', event.data);
						}
					} else {
						console.error('Message is not a string:', typeof event.data);
						messageData = event.data;
					}

					// Use our improved message conversion pipeline
					// First normalize to a consistent format
					const normalizedMessage = normalizeWebSocketMessage(messageData);
					const outputs = normalizedMessage.outputs;
					if (!isSystemUpdate(normalizedMessage)) {
						console.log('Normalized message received from websocket:', normalizedMessage);
					}

					// Check for system updates and flow progress
					if (isSystemUpdate(messageData)) {
						// Assign a shallow copy to ensure reactivity if properties change within the object
						systemUpdateStatus = outputs as SystemUpdate;
						console.debug('Updated system status:', systemUpdateStatus, ' step_name: ', systemUpdateStatus.step_name, ' waiting_on:', systemUpdateStatus.waiting_on);
					} else if (normalizedMessage.type === 'flow_progress_update') {
						// Handle flow progress updates - update system status but don't add to message display
						systemUpdateStatus = {
							source: outputs?.source || 'System',
							step_name: outputs?.step_name || 'Processing',
							status: outputs?.status || 'IN_PROGRESS',
							message: outputs?.message || '',
							timestamp: normalizedMessage.timestamp || new Date().toISOString(),
							waiting_on: outputs?.waiting_on || {}
						};
						console.debug('Updated flow progress status:', systemUpdateStatus);
					} else if (normalizedMessage.type === 'agent_announcement') {
						// Handle agent announcements - update system status but don't add to message display
						const agentId = outputs?.agent_id || 'Unknown';
						const action = outputs?.action || 'update';
						const statusMessage = outputs?.status_message || `Agent ${action}`;

						systemUpdateStatus = {
							source: agentId,
							step_name: statusMessage,
							status:
								action === 'joined' ? 'STARTED' : action === 'left' ? 'COMPLETED' : 'IN_PROGRESS',
							message: statusMessage,
							timestamp: normalizedMessage.timestamp || new Date().toISOString(),
							waiting_on: {}
						};
						console.debug('Updated agent status:', systemUpdateStatus);
					} else {
						// Add the message to the display
						console.debug('added message for display: ', normalizedMessage);
						addMessage(normalizedMessage);
					}
					// Check if this is a system prompt and update state
					if (normalizedMessage.type === 'system_prompt' && outputs) {
						currentUIMessage = outputs as SystemPromptMessage;

						// Determine input type
						const inputs = currentUIMessage.options;

						// Check if it's a selection request
						if (inputs && Array.isArray(inputs) && inputs.length > 0) {
							selectionOptions = inputs.map((val) => String(val));
							isConfirmRequest = false;
						} else {
							selectionOptions = [];
							isConfirmRequest = true;
						}

						console.debug('Updated manager request state:', {
							selectionOptions,
							isConfirmRequest
						});
					}
				} catch (e) {
					console.error('Error processing WebSocket message:', e);
					// Fallback for any errors - display as error message
					if (event.data && typeof event.data === 'string' && event.data.trim() !== '') {
						const errorMsg: Message = {
							timestamp: new Date().toLocaleTimeString(),
							message_id: `error_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`,
							preview: `Error processing message: ${e}\nRaw data: ${event.data}`,
							type: 'system_error' as MessageType
						};
						addMessage(errorMsg);
					}
				}
			};

			socket.onerror = (error) => {
				console.error('WebSocket error:', error);
				connectionError = 'WebSocket connection error. See console for details.';
				isConnected = false;
			};

			socket.onclose = (event) => {
				console.log('WebSocket connection closed:', event.code, event.reason);
				isConnected = false;

				// Check if this is a session termination by the backend
				const isSessionTerminated = event.code === 1000 && event.reason?.includes('TERMINATED');
				const isServerShutdown = event.code === 1001 || event.code === 1006;

				if (isSessionTerminated) {
					console.log('Session terminated by backend, stopping reconnection attempts');
					connectionError = 'Session terminated by server';
					isReconnecting = false;
					return;
				}

				// Set reconnecting state for other types of disconnections
				isReconnecting = true;
				reconnectAttempts++;

				// Update connection error message with close reason
				if (event.code !== 1000) {
					// 1000 is normal closure
					connectionError = `Connection closed. Code: ${event.code}${event.reason ? ', Reason: ' + event.reason : ''}`;
				}

				// Attempt to reconnect after a delay with exponential backoff (but not in readonly/demo mode)
				if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS && !readonly && sessionStatus !== 'demo' && !isServerShutdown) {
					const backoffDelay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000); // Exponential backoff, max 30s
					console.debug(
						`Will attempt reconnection ${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS} in ${backoffDelay}ms`
					);

					reconnectTimeout = setTimeout(() => {
						if (!isConnected && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
							console.debug(
								`Attempting to reconnect... (Attempt ${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})`
							);
							connectWebSocket();
						}
					}, backoffDelay) as unknown as number;
				} else {
					console.error(`Max reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached, giving up`);
					connectionError = `Connection failed after ${MAX_RECONNECT_ATTEMPTS} attempts`;
					isReconnecting = false;
				}
			};
		} catch {
			console.error('Error creating WebSocket:');
			connectionError = `Error creating WebSocket`;
			isConnected = false;
			isReconnecting = false; // Ensure isReconnecting is false on error
		}
	}

	// Send a message to the WebSocket server
	function sendMessage() {
		if (!socket || socket.readyState !== WebSocket.OPEN || !inputMessage.trim()) {
			return;
		}

		console.log('Sending message:', inputMessage);

		// Create user message with unique ID for deduplication
		const messageId = `user_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
		const userMessage = createUserResponse(false, null, inputMessage, false, false, humanInLoop, messageId);

		try {
			socket.send(JSON.stringify(userMessage));

			// Create a properly formatted message object from the user message
			const userMessageFormatted: Message = {
				type: 'user_response',
				message_id: messageId, // Use the same ID for deduplication
				preview: inputMessage,
				timestamp: new Date().toISOString(),
				agent_info: {
					agent_id: 'user',
					agent_name: 'user',
					role: 'user',
					description: 'The user interacting with the terminal.',
					session_id: get(sessionId)
				}
			};

			addMessage(userMessageFormatted); // Display user message immediately
			inputMessage = ''; // Clear input field
		} catch (error) {
			console.error('Error sending message:', error);
			addSystemMessage(`Error sending message: ${error}`);
		}
	}

	// Function to handle restored messages
	export function handleMessage(message: Message) {
		// Add the message directly without WebSocket processing
		addMessage(message);

		// Also add to message stores for consistency
		addToMessageStore(message);

		// Update message history for persistence
		if (get(sessionId)) {
			messageHistory.addMessage(message);
		}
	}

	// Function to send a run_flow request
	export function sendRunFlowRequest(flow: string, dataset: string, record: string, criteria: string) {
		if (!socket || socket.readyState !== WebSocket.OPEN) {
			console.warn('WebSocket not open. Cannot send run flow request.');
			addSystemMessage('Error: Connection not open.');
			return;
		}

		// Validate all required parameters
		if (!flow) {
			console.error('Flow is empty or undefined!');
			addSystemMessage('Error: Please select a flow before running');
			return;
		}

		console.log('Running flow with:', { flow, record_id: record, criteria });

		const data = {
			type: 'run_flow',
			flow: flow,
			record_id: record,
			dataset: dataset || '', // Include dataset parameter
			criteria: criteria || '' // Default to empty string if criteria is not provided
		};

		try {
			socket.send(JSON.stringify(data));
			console.log('Sent run_flow request via WebSocket:', data);
			addSystemMessage(
				`Sent run_flow request for flow '${flow}' with record '${record}' and criteria: ${criteria}`
			);
		} catch (e) {
			console.error('Error sending run_flow request:', e);
			addSystemMessage(`Error sending run_flow request: ${e}`);
		}
	}

	// Function to send a pull_task request
	function sendPullTaskRequest() {
		if (!socket || socket.readyState !== WebSocket.OPEN) {
			console.warn('WebSocket not open. Cannot send pull task request.');
			addSystemMessage('Error: Connection not open.');
			return;
		}

		console.log('Sending pull_task request');

		const data = {
			type: 'pull_task'
		};

		try {
			socket.send(JSON.stringify(data));
			console.log('Sent pull_task request via WebSocket:', data);
			addSystemMessage('Sent pull_task request');
			flowRunning.set(true); // Set flow running state to true
		} catch (e) {
			console.error('Error sending pull_task request:', e);
			addSystemMessage(`Error sending pull_task request: ${e}`);
		}
	}

	// Function to send a pull_tox request
	function sendPullToxRequest() {
		if (!socket || socket.readyState !== WebSocket.OPEN) {
			console.warn('WebSocket not open. Cannot send pull task request.');
			addSystemMessage('Error: Connection not open.');
			return;
		}

		console.log('Sending pull_tox request');

		const data = {
			type: 'pull_tox'
		};

		try {
			socket.send(JSON.stringify(data));
			console.log('Sent pull_tox request via WebSocket:', data);
			addSystemMessage('Sent pull_tox request');
			flowRunning.set(true); // Set flow running state to true
		} catch (e) {
			console.error('Error sending pull_tox request:', e);
			addSystemMessage(`Error sending pull_tox request: ${e}`);
		}
	}

	// Format timestamp - strip milliseconds part
	function formatTimestamp(timestamp: string): string {
		try {
			const date = new Date(timestamp);
			return date.toISOString().replace(/\.\d{3}Z$/, '');
		} catch (e) {
			return timestamp;
		}
	}

	// Add a message to the list and scroll down
	function addMessage(message: Message) {
		// Check for duplicate messages using message_id
		if (message.message_id && displayedMessageIds.has(message.message_id)) {
			console.debug(`Skipping duplicate message: ${message.message_id}`);
			return;
		}

		// Format timestamp (remove fractions of a second)
		if (message.timestamp) {
			message.timestamp = formatTimestamp(message.timestamp);
		}

		// Track this message ID as displayed
		if (message.message_id) {
			displayedMessageIds.add(message.message_id);
		}

		messages = [...messages, message];

		// Add to global message store for sidebar
		addToMessageStore(message);

		// Save to message history store if we have a session ID
		const currentSessionId = get(sessionId);
		if (currentSessionId) {
			messageHistory.saveForSession(currentSessionId, messages);
		}

		// Scroll to bottom after message is added
		setTimeout(() => {
			if (messageListElement) {
				messageListElement.scrollTop = messageListElement.scrollHeight;
			}
		}, 0);
	}

	// Add a system message
	function addSystemMessage(content: string) {
		const systemMessage: Message = {
			type: 'system_message',
			timestamp: new Date().toISOString(),
			preview: content,
			message_id: `system_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`
		};
		addMessage(systemMessage);
	}

	// Handle Enter key press in input field
	function handleKeyDown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}
</script>

<div class="terminal-container terminal-only">
	<!-- Status Bar -->
	<div class="terminal-status-bar">
		<div class="status-left">
			<span class="session-info">Session: {currentSessionId.slice(0, 8)}...</span>
			<span class="status-text status-{displayStatus}">
				{#if displayStatus === 'running'}
					active
				{:else if displayStatus === 'completed'}
					completed
				{:else if displayStatus === 'failed'}
					failed
				{:else if displayStatus === 'idle'}
					idle
				{:else if displayStatus === 'disconnected'}
					disconnected
				{:else if displayStatus === 'reconnecting'}
					reconnecting...
				{:else if displayStatus === 'terminated'}
					terminated
				{:else}
					{displayStatus}
				{/if}
			</span>
			{#if !isResumable}
				<span class="readonly-text">read-only</span>
			{/if}
		</div>
		<div class="status-right">
			<!-- Token usage display -->
			{#if $tokenUsageDisplay.messageCount > 0}
			<div class="token-usage">
				<span class="token-count">{$tokenUsageDisplay.totalTokens} tokens</span>
				<span class="token-cost">{$tokenUsageDisplay.cost}</span>
			</div>
			{/if}
			<div>
				<button class="text-button" onclick={() => (window.location.href = '/terminal')}>
					[ new session ]
				</button>
			</div>
		</div>
	</div>

	<!-- Message Display Area -->
	<div class="console" id="console-messages" bind:this={messageListElement}>
		{#each messages as msg (msg.message_id)}
			<MessageDisplay message={msg} />
		{/each}
	</div>

	<!-- Manager Request Area - Only visible when there's an active request -->
	{#if currentUIMessage}
		<div class="manager-request-area">
			<div class="manager-request-content">
				<span class="request-tag">[REQUEST]</span>
				{#if currentUIMessage.content}
					<span class="request-text">{currentUIMessage.content}</span>
				{/if}
			</div>
		</div>
	{/if}

	<!-- Input Area -->
	<div class="input-area mt-3">
		<textarea
			class="terminal-input"
			bind:value={inputMessage}
			onkeydown={handleKeyDown}
			placeholder="Enter command or message..."
			rows="1"
			disabled={!isConnected}
		></textarea>

		<!-- Terminal-style Buttons with Dynamic Manager Request Controls -->
		<div class="terminal-buttons">
			<!-- Manager request selection options - only shown when there's a selection request -->
			{#if currentUIMessage && selectionOptions.length > 0}
				<div class="selection-options">
					Select:
					{#each selectionOptions as option}
						<button
							type="button"
							class="terminal-button option-button"
							onclick={() => handleSelection(option)}
						>
							[ {option} ]
						</button>
					{/each}
				</div>
			{:else if currentUIMessage}
				<!-- Confirm/Reject buttons for manager requests -->
				<button
					type="button"
					class="terminal-button confirm-button"
					onclick={() => handleConfirm(true)}
				>
					[ Confirm ]
				</button>
				<button
					type="button"
					class="terminal-button reject-button"
					onclick={() => handleConfirm(false)}
				>
					[ Reject ]
				</button>
				<button type="button" class="terminal-button halt-button" onclick={handleHalt}>
					[ Halt ]
				</button>
			{/if}
			<!-- Standard message buttons that are always active -->
			<button
				type="button"
				class="terminal-button submit-button"
				onclick={sendMessage}
				disabled={!isConnected || !inputMessage.trim()}
			>
				[ Submit ]
			</button>
			<button
				type="button"
				class="terminal-button interrupt-button"
				aria-label="Interrupt"
				onclick={toggleInterrupt}
				title={isInterruptEnabled ? 'Resume Flow' : 'Interrupt'}
				disabled={!isConnected}
			>
				<i class="bi {isInterruptEnabled ? 'bi-play-circle-fill' : 'bi-pause-circle-fill'}"></i>
				{isInterruptEnabled ? '[ Resume ]' : '[ Interrupt ]'}
			</button>
			<!-- Auto Approve Toggle Switch -->
			<button
				type="button"
				class="terminal-button approve-button"
				aria-label="Auto Approve"
				onclick={toggleAutoApprove}
				title={humanInLoop ? 'Disable Auto Approve' : 'Enable Auto Approve'}
				disabled={!isConnected}
			>
				<i class="bi {humanInLoop ? 'bi-toggle-off' : 'bi-toggle-on'}"></i>
				{humanInLoop ? '[ human in loop ]' : '[ let it ride ]'}
			</button>

			<button
				type="button"
				class="terminal-button pull-task-button"
				aria-label="Pull Task"
				onclick={sendPullTaskRequest}
				title="Pull Task"
				disabled={!isConnected || $flowRunning}
			>
				<i class="bi bi-cloud-download"></i>
				[ pull task ]
			</button>

			<button
				type="button"
				class="terminal-button pull-task-button"
				aria-label="Pull Task"
				onclick={sendPullToxRequest}
				title="Pull Task"
				disabled={!isConnected || $flowRunning}
			>
				<i class="bi bi-cloud-download"></i>
				[ tox example ]
			</button>
		</div>
	</div>
</div>
