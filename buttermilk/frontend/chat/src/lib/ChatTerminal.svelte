<script lang="ts">
import { onDestroy, onMount } from 'svelte';
import { get } from 'svelte/store';
import MessageDisplay from './components/MessageDisplay.svelte';
import { flowRunning, selectedFlow as flowStore, selectedRecord as recordStore, selectedCriteria, selectedModel } from './stores/apiStore';
import { messageHistory } from './stores/messageHistoryStore';
import { addMessage as addToMessageStore } from './stores/messageStore';
import { sessionId } from './stores/sessionStore';
import './styles/terminal.scss';
import {
	type ManagerMessage,
	type ManagerResponse,
	type Message,
	type MessageType,
	type SystemUpdate,
	createManagerResponse, isSystemUpdate,
	normalizeWebSocketMessage
} from './utils/messageUtils';


  // WebSocket connection parameters
  export let wsUrl: string;
  export let selectedFlow: string; // Prop for selected flow
  export let selectedRecord: string; // Prop for selected record
  
  // Monitor props for debugging
  $: {
    console.debug(`ChatTerminal props updated: selectedFlow='${selectedFlow}', selectedRecord='${selectedRecord}'`);
  }
  
  // Component state
  let socket: WebSocket | null = null;
  let messages: Message[] = [];
  let inputMessage = '';
  let isConnected = false;
  let connectionError = '';
  let isInterruptEnabled = false;
  let humanInLoop = true;
  let messageListElement: HTMLDivElement; // To control scrolling
  export let selectedTheme = 'theme-term'; // Default theme
  let isReconnecting = false; // Track reconnection attempts
  let reconnectAttempts = 0; // Count reconnection attempts
  
  // System update state
  let systemUpdateStatus: SystemUpdate | null = null;
  
  // Manager request state
  let currentUIMessage: ManagerMessage | null = null;
  let selectionOptions: string[] = [];
  let isConfirmRequest = false;

  // Load stored messages for the current session
  function loadStoredMessages(sessionIdToLoad: string) {
    try {
      const key = `messageHistory_${sessionIdToLoad}`;
      const stored = localStorage.getItem(key);
      if (stored) {
        const storedMessages: Message[] = JSON.parse(stored);
        console.debug(`Loading ${storedMessages.length} stored messages for session ${sessionIdToLoad}`);
        
        // Set the messages array directly to avoid any reprocessing
        messages = storedMessages;
        
        // Restore manager request state from the last ui_message if any
        const lastUIMessage = storedMessages.slice().reverse().find(msg => msg.type === 'ui_message');
        if (lastUIMessage && lastUIMessage.outputs) {
          currentUIMessage = lastUIMessage.outputs as ManagerMessage;
          
          // Determine input type
          const inputs = currentUIMessage.options;
          
          // Check if it's a selection request
          if (inputs && Array.isArray(inputs) && inputs.length > 0) {
            selectionOptions = inputs.map(val => String(val));
            isConfirmRequest = false;
          } else {
            selectionOptions = [];
            isConfirmRequest = true;
          }

          console.debug("Restored manager request state from stored messages:", {
            selectionOptions,
            isConfirmRequest,
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
  onMount(async () => { // Make the callback async
    console.debug('Attempting direct WebSocket connection to:', wsUrl);
    // Ensure we have a session ID before attempting to connect
    if (!get(sessionId)) {
      addSystemMessage('Initializing session...');
      await getNewSessionId(); // Fetches and stores a clean session ID
    }
    if (get(sessionId)) {
      // Load stored messages for this session
      loadStoredMessages(get(sessionId));
      
      // wsUrl prop should be the base like "ws://localhost:5173/ws"
      console.debug('Attempting direct WebSocket connection. Base wsUrl prop:', wsUrl);
      connectWebSocket();
    } else {
      const errorMsg = "Failed to obtain session ID on mount. WebSocket connection not established.";
      connectionError = errorMsg;
      addSystemMessage(`Error: ${errorMsg}`);
      console.error(errorMsg);
    }

    document.body.className = selectedTheme; // Apply default theme to body
  });
  
  // Handle component destruction
  onDestroy(() => {
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
    selectedModel.set('');
    
    // Clear messages
    messages = [];
    // Get new session ID
    addSystemMessage('Starting a new session...');
    await getNewSessionId();
    // Reconnect
    connectWebSocket();
  }

  // Toggle auto-approve state
  function toggleAutoApprove() {
    humanInLoop = !humanInLoop;
    console.log("Toggled human in loop state:", humanInLoop, " approving:", !humanInLoop);
    const message = createManagerResponse(!humanInLoop, null, null, null, null, humanInLoop);
    sendManagerResponse(message);
  }

  // Toggle interrupt state
  function toggleInterrupt() { 
    isInterruptEnabled = !isInterruptEnabled;

    let message;
    if (isInterruptEnabled) {
      message = {
        type: "TaskProcessingStarted",
        role: "MANAGER",
        agent_id: "web socket"
      };
      // Also send separate interrupt message
      const response = createManagerResponse(false, null, null, null, true, humanInLoop);
      sendManagerResponse(response);
      
    } else {
      message = {
        type: "TaskProcessingComplete",
        role: "MANAGER",
        agent_id: "web socket"
      };
      const response = createManagerResponse(false, null, null, null, false, humanInLoop);
      sendManagerResponse(response);

    }

    try {
      console.log("Sending interrupt message:", message);
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(message));
      }
    } catch (e) {
      console.error("Error sending interrupt start message:", e);
    }
  }

  // Function to send ManagerResponse back via WebSocket
  function handleManagerResponse(event: CustomEvent<ManagerResponse>) {
    const response = event.detail;
    console.debug("Received manager response from component:", response);
    sendManagerResponse(response);
  }

  function sendManagerResponse(response: ManagerResponse) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      try {
        socket.send(JSON.stringify(response));
        console.log("Sent manager response via WebSocket:", response);
        // Clear current manager request after responding
        currentUIMessage = null;
        selectionOptions = [];
        isConfirmRequest = false;
      } catch (e) {
        console.error("Error sending manager response:", e);
        addSystemMessage(`Error sending response: ${e}`);
      }
    } else {
      console.error("WebSocket not open, cannot send manager response.");
      addSystemMessage("Error: Connection not open.");
    }
  }
  
  // New function to create manager response


  // Handle selection from options
  function handleSelection(value: string) {
    const response = createManagerResponse(true, value, null, null, false, humanInLoop);
    sendManagerResponse(response);
  }
  
  // Handle confirm/reject
  function handleConfirm(value: boolean) {
    const response = createManagerResponse(value, null, null, null, false, humanInLoop);
    sendManagerResponse(response);
  }
  
  // Handle halt
  function handleHalt() {
    const response = createManagerResponse(false, null, null, true, null, humanInLoop);
    sendManagerResponse(response);
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
      const data = await response.json() as SessionResponse;
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
    let wsUrlWithSession;
    try {
        
      wsUrlWithSession = `${wsUrl}/${currentSessionId}`;
      console.debug('Attempting to connect to WebSocket with URL:', wsUrlWithSession);
      
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

      socket = new WebSocket(wsUrlWithSession);
      
      socket.onopen = () => {
        console.log('Direct WebSocket connection established');
        clearTimeout(connectionTimeout);
        isConnected = true;
        connectionError = '';
        isReconnecting = false; // Reset reconnection state
        reconnectAttempts = 0; // Reset reconnection counter
        // Connection status now appears in UI header instead of as system message
      };
      
      socket.onmessage = (event) => {
        try {
          // First try to parse the message as JSON
          let messageData: unknown;
          
          if (typeof event.data === 'string') {
            try {
              messageData = JSON.parse(event.data);
              console.debug('Message received:', messageData);
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
          } else {
            console.debug('Normalized message received from websocket:', normalizedMessage);
          }

          // Check for system updates
          if (isSystemUpdate(messageData)) {
            // Assign a shallow copy to ensure reactivity if properties change within the object
            systemUpdateStatus = outputs as SystemUpdate;
            console.debug('Updated system status:', systemUpdateStatus);  
            // Add logs to check specific properties
            console.debug('systemUpdateStatus.step_name:', systemUpdateStatus.step_name);
            console.debug('systemUpdateStatus.waiting_on:', systemUpdateStatus.waiting_on);
          
          } else {
            // Add the message to the display
            addMessage(normalizedMessage);
          }
          // Check if this is a manager request and update state
          if (normalizedMessage.type === 'ui_message' && outputs) {
            currentUIMessage = outputs as ManagerMessage;
            
            // Determine input type
            const inputs = currentUIMessage.options;
            
            // Check if it's a selection request
            if (inputs && Array.isArray(inputs) && inputs.length > 0) {
              selectionOptions = inputs.map(val => String(val));
              isConfirmRequest = false;
            } 
            else {
              selectionOptions = [];
              isConfirmRequest = true;
            }
            
            console.debug("Updated manager request state:", { 
              selectionOptions, 
              isConfirmRequest, 
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
              type: 'system_error' as MessageType,
            };
            addMessage(errorMsg);
          }
        }
      };
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        connectionError = 'WebSocket connection error. See console for details.';
        // isConnected = false;
      };
      
      socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        isConnected = false;
        
        // Set reconnecting state
        isReconnecting = true;
        reconnectAttempts++;
        
        // Update connection error message with close reason
        if (event.code !== 1000) { // 1000 is normal closure
          connectionError = `Connection closed. Code: ${event.code}${event.reason ? ', Reason: ' + event.reason : ''}`;
        }
        
        // Attempt to reconnect after a delay
        setTimeout(() => {
          if (!isConnected) {
            console.debug(`Attempting to reconnect... (Attempt ${reconnectAttempts})`);            connectWebSocket();
          }
        }, 5000);
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      connectionError = `Error creating WebSocket: ${error}`;
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
    
    // Create user message
    const userMessage = createManagerResponse(false, null, inputMessage, false, false, humanInLoop);
    
    try {
      socket.send(JSON.stringify(userMessage));
      
      // Create a properly formatted message object from the user message
      const userMessageFormatted: Message = {
        type: 'user',
        message_id: `user_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`,
        preview: inputMessage,
        timestamp: new Date().toISOString(),
      };
      
      addMessage(userMessageFormatted); // Display user message immediately
      inputMessage = ''; // Clear input field
    } catch (e) {
      console.error('Error sending message:', e);
      addSystemMessage(`Error sending message: ${e}`);
    }
  }
    
  // Function to send a run_flow request
  export function sendRunFlowRequest(flow: string, record: string, criteria: string) {
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
      type: "run_flow",
      flow: flow,
      record_id: record,
      criteria: criteria || '' // Default to empty string if criteria is not provided
    };

    try {
      socket.send(JSON.stringify(data));
      console.log('Sent run_flow request via WebSocket:', data);
      addSystemMessage(`Sent run_flow request for flow '${flow}' with record '${record}' and criteria: ${criteria}`);
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
      type: "pull_task"
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
      type: "pull_tox"
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
    // Format timestamp (remove fractions of a second)
    if (message.timestamp) {
      message.timestamp = formatTimestamp(message.timestamp);
    }
    
    // Note: Manager request state is handled in WebSocket message handler
     
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
      message_id: `system_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`,
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
    <!-- Header with Connection Status, System Status, and Controls -->
    <div class="header-controls">
      <div class="status-container">
        <div class="connection-status {isConnected ? 'connected' : isReconnecting ? 'reconnecting' : 'disconnected'}">
          <span class="status-indicator {isReconnecting ? 'spinner' : ''}"></span>
          {isConnected ? 'Connected' : isReconnecting ? `Reconnecting${reconnectAttempts > 1 ? ' (' + reconnectAttempts + ')' : ''}` : 'Disconnected'}
          {#if !isConnected && !isReconnecting && connectionError}
            <span class="error-text"> - {connectionError}</span>
          {/if}
        </div>
        
        <!-- System update status display -->
        {#if systemUpdateStatus }
          <div class="system-status systemUpdateStatus.status}">
            <span class="system-step">{systemUpdateStatus.step_name}</span>
            {#if systemUpdateStatus.waiting_on}
              <span class="waiting-on">
                Waiting on: {Object.keys(systemUpdateStatus.waiting_on).join(', ')}
              </span>
            {/if}
          </div>
        {/if}
      </div>
      
      <div>
        <button class="terminal-button new-session-button" onclick={startNewSession} title="Start New Session">
          <i class="bi bi-arrow-clockwise"></i> new session
        </button>
      </div>
    </div>

    <!-- Message Display Area -->
    <div class="console" id="console-messages"  bind:this={messageListElement}>
      {#each messages as msg }
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
                  onclick={() => handleSelection(option)}>
                  [ {option} ]
                </button>
              {/each}
            </div>
          {:else if currentUIMessage}
            <!-- Confirm/Reject buttons for manager requests -->
            <button 
              type="button" 
              class="terminal-button confirm-button"
              onclick={() => handleConfirm(true)}>
              [ Confirm ]
            </button>
            <button 
              type="button" 
              class="terminal-button reject-button"
              onclick={() => handleConfirm(false)}>
              [ Reject ]
            </button>
            <button 
              type="button" 
              class="terminal-button halt-button"
              onclick={handleHalt}>
              [ Halt ]
            </button>
            {/if}
            <!-- Standard message buttons that are always active -->
            <button 
              type="button" 
              class="terminal-button submit-button"
              onclick={sendMessage}
              disabled={!isConnected || !inputMessage.trim()}>
              [ Submit ]
            </button>
            <button 
              type="button" 
              class="terminal-button interrupt-button"
              aria-label="Interrupt"
              onclick={toggleInterrupt}
              title={isInterruptEnabled ? 'Resume Flow' : 'Interrupt'}
              disabled={!isConnected}>
              
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
              disabled={!isConnected}>
              
              <i class="bi {humanInLoop ? 'bi-toggle-off': 'bi-toggle-on'}"></i>
              {humanInLoop ? '[ human in loop ]': '[ let it ride ]'}
            </button>
            
            <button 
              type="button" 
              class="terminal-button pull-task-button"
              aria-label="Pull Task"
              onclick={sendPullTaskRequest}
              title="Pull Task"
              disabled={!isConnected || $flowRunning}>
                  
              <i class="bi bi-cloud-download"></i>
              [ pull task ]
            </button>
            
            <button 
              type="button" 
              class="terminal-button pull-task-button"
              aria-label="Pull Task"
              onclick={sendPullToxRequest}
              title="Pull Task"
              disabled={!isConnected || $flowRunning}>
                  
              <i class="bi bi-cloud-download"></i>
              [ tox example ]
            </button>
        </div>
      </div>
</div>
