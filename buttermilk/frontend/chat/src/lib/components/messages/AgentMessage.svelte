<script lang="ts">
	import { type GenericMessage } from '$lib/utils/messageUtils';
	import BasicMessage from './BasicMessage.svelte';

	// Note: The component receives the 'Message' type from convertToDisplayMessage,
	// where the original UIMessage is nested within message.outputs
	export let message: GenericMessage;

	// --- Reactive variables to access nested data and determine input type ---

	// Extract the original UIMessage data (cast for type safety)
	$: MessageData = message.outputs as GenericMessage;

	$: agentName = message.agent_info?.agent_id || 'SYSTEM';
	$: model = message.agent_info?.parameters?.model;
	$: template = message.agent_info?.parameters?.template;
</script>

<div class="message-terminal agent-message">
	<BasicMessage {message}>
		<svelte:fragment slot="agentNick">[{agentName}]</svelte:fragment>
		<div class="message-text col-sm-10">
			<span class="message-body agent-text">
				<slot name="messageContent">
					{MessageData.content}
				</slot>
			</span>
		</div>
	</BasicMessage>
</div>

<style>
	.agent-name {
		color: #5bdeac;
	}

	.agent-text {
		color: #eefaec;
	}
</style>
