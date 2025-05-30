<script lang="ts">
  import {
  	type ManagerResponse,
  	type Message,
  	type UIMessage
  } from '$lib/utils/messageUtils';
  import { createEventDispatcher } from 'svelte';
  import BasicMessage from './BasicMessage.svelte';

    // Props
  // Note: The component receives the 'Message' type from convertToDisplayMessage,
  // where the original UIMessage is nested within message.outputs
  export let message: Message; 

  // Dispatcher for sending the response back
  const dispatch = createEventDispatcher<{ managerResponse: ManagerResponse }>();

  // --- Reactive variables to access nested data and determine input type ---
  
  // Extract the original UIMessage data (cast for type safety)
  $: UIMessageData = message.outputs as UIMessage | undefined; 

  $: agentName = message.agent_info?.agent_id || 'SYSTEM';

</script>

<div class="message-terminal manager-message">
  <BasicMessage message={message}>
    <span class="agent-nick manager-name">
      <slot name="agentNick">[{agentName}]</slot>
    </span>
    <div class="message-text col-sm-10">
      <span class="message-body manager-text">
        <slot name="messageContent">
          {message.outputs.content}
        </slot>
      </span>
    </div>
  </BasicMessage>
</div>

<style>
  .manager-name {
    color: #5bc0de;
  }
  
  .manager-text {
    color: #f0f0f0;
  }
  
</style>
