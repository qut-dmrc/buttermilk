<script lang="ts">
  import { getModelColor, type Message } from '$lib/utils/messageUtils';
  import { marked } from 'marked';
  import { slide } from 'svelte/transition';
// Import common expandable details styles
  import '$lib/styles/expandable-details.scss';
  import BasicMessage from './BasicMessage.svelte';

  // Props
  export let message: Message;
  export let expanded = false;

  // Local state
  let isLiteratureExpanded = false;
  
  // Parse markdown content with newline fixing and citation highlighting
  function parseMarkdown(content: string): string {
    if (!content) return 'No content available';
    
    // Replace escaped newlines with actual newlines
    let cleanedContent = content.replace(/\\n/g, '\n');
    
    // Highlight citations with regex - matches patterns like (Author Year) or (Author et al., 2023)
    // Be overly inclusive to catch most citation formats
    cleanedContent = cleanedContent.replace(/\(([^)]{2,40}?(?:\d{4}|20\d{2})[^)]{0,20}?)\)/g, 
      (match, citation) => `**${match}**`);
    
    try {
      // Use marked.parse synchronously to ensure we return a string
      return marked.parse(cleanedContent, { async: false }) as string;
    } catch (e) {
      console.error('Error parsing markdown:', e);
      return cleanedContent;
    }
  }
  
  // Get researcher data from message outputs
  $: researcherData = message.outputs || {};
  $: literature = researcherData.literature || [];
  $: response = researcherData.response || '';
  $: modelBasedColor = getModelColor(message.agent_info?.parameters?.model);

  // Toggle literature expansion
  function toggleLiterature() {
    isLiteratureExpanded = !isLiteratureExpanded;
  }
</script>

<div class="message-terminal" style="color: {modelBasedColor}">
  <BasicMessage message={message}>
    <svelte:fragment slot="agentNick">[RESEARCH]</svelte:fragment>
    
    <svelte:fragment slot="messagePrefix">
      <i class="bi bi-book"></i>{literature.length} refs
    </svelte:fragment>
    
    <svelte:fragment slot="messageContent">
      <!-- Main response section -->
      <div class="content-inline">
        {@html parseMarkdown(response)}

        <!-- Literature Toggle (Inline button) -->
        <button 
        class="content-toggle-inline" 
        on:click={toggleLiterature}
        title={isLiteratureExpanded ? "Hide literature" : "View literature"}
      >
        {isLiteratureExpanded ? '[-]' : '[+]'} references
      </button>
      </div>
      
    </svelte:fragment>
    
    <svelte:fragment slot="messageExpanded">
      <!-- Expandable literature section -->
      {#if isLiteratureExpanded && literature.length > 0}
      <div class="literature">
          <div class="expanded-content markdown-content" transition:slide={{ duration: 200 }}>
            <ol>
              {#each literature as item}
                <li>
                  <div class="literature-summary">{@html parseMarkdown(item.summary)}</div>
                  <div class="literature-citation">{@html parseMarkdown(item.citation)}</div>
                </li>
              {/each}
            </ol>
          </div>
        </div>
      {/if}
    </svelte:fragment>
  </BasicMessage>
</div>

<!-- Sidebar Card (only if this message should appear in sidebar) -->
{#if expanded}
  <div class="sidebar-card mt-2 mb-3">
    <div class="card">
      <div class="card-header d-flex justify-content-between align-items-center" 
           style="background-color: rgba(95, 173, 170, 0.1); border-color: #5fadaa; border-left: 3px solid #5fadaa;">
        <span>
          <span class="agent-tag">RESEARCH</span> <small>{message.agent_info?.agent_name || 'Researcher'}</small>
        </span>
        <span class="badge bg-light text-dark">
          {literature.length} references
        </span>
      </div>
      
      <div class="card-body p-2">
        <div class="tiny-text">
          <div class="text-truncate">{response.substring(0, 100)}...</div>
          <div><small>{literature.length} literature reference{literature.length !== 1 ? 's' : ''}</small></div>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .literature li {
    font-size: 0.7rem;
  }
  .literature-summary {
    margin-bottom: 2px;
    font-size: 0.7rem;
  }
  
  .literature-citation {
    color: #bbbbbb;
    font-size: 0.7rem;
  }
  
</style>
