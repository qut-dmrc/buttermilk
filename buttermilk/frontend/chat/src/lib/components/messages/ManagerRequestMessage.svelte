<script lang="ts">
  import {
  	type ManagerResponse,
  	type Message,
  	type UIMessage
  } from '$lib/utils/messageUtils';
  import { createEventDispatcher } from 'svelte';

  import { createManagerResponse } from '$lib/utils/messageUtils';
  // Props
  // Note: The component receives the 'Message' type from convertToDisplayMessage,
  // where the original UIMessage is nested within message.outputs
  export let message: Message; 

  // Dispatcher for sending the response back
  const dispatch = createEventDispatcher<{ managerResponse: ManagerResponse }>();

  // --- Reactive variables to access nested data and determine input type ---
  
  // Extract the original UIMessage data (cast for type safety)
  $: UIMessageData = message.outputs as UIMessage | undefined; 

  // Local state for user input
  let userPrompt: string = '';
  let selectedValue: string | boolean | null = null; // Can hold boolean for confirm, string for selection/text

  // Function to send a response
  function sendResponse(confirm: boolean, halt: boolean = false, interrupt: boolean = false, human_in_loop: boolean = true) {
    const response: ManagerResponse = createManagerResponse(
      confirm,
      typeof selectedValue === 'string' ? selectedValue : null,
      userPrompt || null,
      halt,
      interrupt,
      human_in_loop,
    );
    dispatch('managerResponse', response);
    console.log("Sent selection response:", response);

  } // End of sendResponse function

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
