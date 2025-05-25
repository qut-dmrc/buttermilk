<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { 
    type UIMessage, 
    type ManagerResponse, 
    type Message // Import the base Message type
  } from '$lib/utils/messageUtils';
  import { processLinksInText } from '$lib/utils/linkUtils';

  // Props
  // Note: The component receives the 'Message' type from convertToDisplayMessage,
  // where the original UIMessage is nested within message.outputs
  export let message: Message; 

  // Dispatcher for sending the response back
  const dispatch = createEventDispatcher<{ managerResponse: ManagerResponse }>();

  // --- Reactive variables to access nested data and determine input type ---
  
  // Extract the original UIMessage data (cast for type safety)
  $: UIMessageData = message.outputs as UIMessage | undefined; 
  $: inputs = UIMessageData?.options;

  // Determine input type based on the 'inputs' field
  $: isSelection = Array.isArray(UIMessageData?.options) && UIMessageData.options.length > 0; // Check if options are provided
  $: isConfirm = typeof UIMessageData?.confirm === 'boolean' && !isSelection; // Only confirm if not selection
  // $: isParameters = typeof inputs?.parameters === 'object' && inputs.parameters !== null && !isSelection && !isConfirm; // Future: Handle parameters
  $: isFreeText = !isSelection && !isConfirm; // Fallback to free text if no specific input defined

  // Process message content for links if any
  $: processedContent = UIMessageData?.content ? processLinksInText(UIMessageData.content) : '';

  // Local state for user input
  let userPrompt: string = '';
  let selectedValue: string | boolean | null = null; // Can hold boolean for confirm, string for selection/text

  // Function to send a response
  function sendResponse(confirm: boolean, halt: boolean = false, interrupt: boolean = false) {
    const response: ManagerResponse = {
      type: 'manager_response',
      confirm: confirm,
      halt: halt,
      interrupt: interrupt,
      content: userPrompt || null, // Send free text input if any
      selection: typeof selectedValue === 'string' ? selectedValue : null, // Send selection if it's a string
      // params: parametersInput, // Future: Add parameters input
    };
    dispatch('managerResponse', response);

    // Optionally clear input after sending
    userPrompt = '';
    selectedValue = null;
  } // End of sendResponse function

  // --- Input Handlers ---

  function handleConfirm(value: boolean) {
    selectedValue = value;
    // If in free text mode and we have text input, use it
    if (isFreeText && userPrompt.trim()) {
      sendResponse(value);
    } else {
      // Otherwise just send the confirmation value
      sendResponse(value);
    }
  }

  function handleSelection(selectionValue: any) {
    // The actual value sent might be different from the display label
    // For now, assume the value itself is what we send.
    selectedValue = String(selectionValue); 
    sendResponse(true); // Selecting an option implies confirmation
  }

  // This function is now only used when pressing Enter in the text field
  function handleFreeTextSubmit() {
    if (userPrompt.trim()) {
      sendResponse(true);
    }
  }

  function handleHalt() {
    sendResponse(false, true); // Halt implies non-confirmation
  }

  function handleInterrupt() {
    sendResponse(false, false, true); // Interrupt implies non-confirmation
  }
</script>

{#if UIMessageData}
    <!-- OTHER Role Request - Displayed as out-of-band trace -->
    <span class="agent-name manager-name">
      <i class="bi bi-gear-fill"></i>Host:
    </span>
    
    <span class="manager-inline">
      <span class="request-tag">[REQUEST]</span> 
        <span class="oob-request-description">{UIMessageData.content}</span>
    </span>
{/if}

<style>
  .manager-name {
    color: #5bc0de;
  }
  
  .manager-inline {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    color: #f0f0f0;
    max-width: 100%;
  }
  
  /* Styling for manager requests addressed to MANAGER role - conversational */
  .conversational-request {
    color: #f0f0f0;
    margin-right: 8px;
  }
  
  /* Styling for out-of-band requests addressed to other roles */
  .request-tag {
    color: #5bc0de;
    font-weight: bold;
    margin-right: 8px;
  }
  
  .oob-request-description {
    color: #bbbbbb;
    font-style: italic;
    margin-right: 8px;
    font-size: 0.95em;
  }
  
  .request-content {
    font-weight: bold;
    margin-right: 8px;
    color: #f0f0f0;
    display: block;
    margin-top: 2px;
    margin-left: 24px;
  }
  
  .request-description {
    color: #bbbbbb;
    display: block;
    margin-right: 8px;
  }
  
  .controls-line {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
    padding-left: 24px;
    align-items: center;
  }
  
  .terminal-input-field {
    background-color: rgba(0, 0, 0, 0.3);
    border: 1px solid #444;
    color: #f0f0f0;
    padding: 2px 6px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    min-width: 200px;
    flex-grow: 1;
  }
  
  .terminal-option {
    background: transparent;
    border: none;
    color: #dddddd;
    cursor: pointer;
    padding: 1px 4px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    transition: color 0.2s;
  }
  
  .terminal-option:hover {
    color: #ffffff;
  }
  
  .terminal-option.confirm:hover {
    color: #5cb85c;
  }
  
  .terminal-option.reject:hover, 
  .terminal-option.halt:hover {
    color: #d9534f;
  }
  
  .terminal-option.submit:hover {
    color: #5bc0de;
  }
  
  .terminal-option.interrupt:hover {
    color: #f0ad4e;
  }
  
  .error-text {
    color: #d9534f;
  }
</style>
