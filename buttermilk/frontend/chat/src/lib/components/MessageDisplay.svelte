<script lang="ts">
  import {
  	type ManagerResponse,
  	type Message,
  	getAgentStyle
  } from '$lib/utils/messageUtils';
  import { createEventDispatcher } from 'svelte';
// Import modular components
  import JudgeMessage from './messages/JudgeMessage.svelte'; // Import JudgeMessage directly
  import AssessmentMessage from './messages/AssessmentMessage.svelte';
  import BasicMessage from './messages/BasicMessage.svelte';
  import * as DifferencesMessageComponent from './messages/DifferencesMessage.svelte';
  import ManagerRequestMessage from './messages/ManagerRequestMessage.svelte';
  import * as RecordMessageComponent from './messages/RecordMessage.svelte';
  import * as ResearcherMessageComponent from './messages/ResearcherMessage.svelte';
  import * as SummaryMessageComponent from './messages/SummaryMessage.svelte';

  const RecordMessage = RecordMessageComponent.default;
  const SummaryMessage = SummaryMessageComponent.default;
  const ResearcherMessage = ResearcherMessageComponent.default;
  const DifferencesMessage = DifferencesMessageComponent.default;

  // Props
  export let message: Message;
  export let expanded = false;

  $: messageType = message.type;

  // Get agent styling using agent_info if available
  $: agentStyle = getAgentStyle(message.agent_info?.agent_name || 'System');

  // Forward the managerResponse event
  const dispatch = createEventDispatcher();

  // Explicitly type the event parameter
  function forwardManagerResponse(event: CustomEvent<ManagerResponse>) {
    dispatch('managerResponse', event.detail);
  }
</script>

{#if messageType === 'judge_reasons'}
  <JudgeMessage {message} {expanded} />
{:else if messageType === 'assessments'}
  <AssessmentMessage {message} {expanded} />
{:else if messageType === 'record'}
  <RecordMessage {message} {expanded} />
{:else if messageType === 'research_result'}
  <ResearcherMessage {message} {expanded} />
{:else if messageType === 'ui_message'}
  <ManagerRequestMessage {message} on:managerResponse={forwardManagerResponse} />
{:else if messageType === 'summary_result'}
  <SummaryMessage {message} {expanded} />
{:else if messageType === 'differences'}
  <DifferencesMessage {message} {expanded} />
{:else if messageType === 'system_update'}
  <BasicMessage {message} {expanded} />
{:else if messageType === 'system_error'}
  <BasicMessage {message} {expanded} />
{:else }
  <BasicMessage {message} {expanded} />
{/if}