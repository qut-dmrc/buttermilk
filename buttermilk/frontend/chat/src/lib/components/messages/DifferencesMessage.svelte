<script lang="ts">
  import { onMount } from 'svelte';
  import { slide } from 'svelte/transition';
  import { getAgentStyle, getAgentEmoji, type Message, getModelIdentifier } from '$lib/utils/messageUtils';
  import BasicMessage from './BasicMessage.svelte';
  
  // Import common expandable details styles
  import '$lib/styles/expandable-details.scss';

  // Props
  export let message: Message;
  export let expanded = false;

  // Local state
  let isContentVisible = false;
  let tooltipElements: HTMLElement[] = [];
  let Tooltip: any;

  // Differences data structure
  interface Expert {
    name: string;
    answer_id: string;
  }

  interface Position {
    experts: Expert[];
    position: string;
  }

  interface Divergence {
    topic: string;
    positions: Position[];
  }

  interface DifferencesData {
    conclusion: string;
    divergences: Divergence[];
  }

  // Extract data from message outputs
  $: differencesData = message.outputs as DifferencesData;  
  // Derive short ID and name
  $: shortAgentId = (message.agent_info?.agent_id ?? 'System' as string).toUpperCase();
  $: modelName = getModelIdentifier(message) || '';


  // Lifecycle
  onMount(() => {
    // Only load Bootstrap in browser environment
    if (typeof window !== 'undefined') {
      import('bootstrap').then(bootstrap => {
        Tooltip = bootstrap.Tooltip;
        initTooltips();
      }).catch(e => {
        console.warn('Bootstrap not available:', e);
      });
    }

    return () => {
      // Destroy tooltips on component unmount
      if (Tooltip && tooltipElements.length > 0) {
        tooltipElements.forEach(el => {
          const tooltip = Tooltip.getInstance(el);
          if (tooltip) tooltip.dispose();
        });
      }
    };
  });

  // Initialize Bootstrap tooltips
  function initTooltips() {
    if (!Tooltip || typeof document === 'undefined') return;
    
    // Small delay to ensure DOM is ready
    setTimeout(() => {
      tooltipElements = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
      tooltipElements.forEach(el => {
        try {
          new Tooltip(el, {
            html: true,
            trigger: 'hover'
          });
        } catch (e) {
          console.warn('Error initializing tooltip:', e);
        }
      });
    }, 100);
  }

  // Toggle content visibility
  function toggleContent() {
    isContentVisible = !isContentVisible;
  }

  // Helper to count experts in a position
  function countExperts(position: Position): number {
    return position.experts.length;
  }

  // Helper to display a truncated list of experts
  function formatExpertsList(experts: Expert[], maxToShow: number = 3): string {
    if (experts.length <= maxToShow) {
      return experts.map(e => e.name).join(", ");
    } else {
      const shownExperts = experts.slice(0, maxToShow);
      return `${shownExperts.map(e => e.name).join(", ")} +${experts.length - maxToShow} more`;
    }
  }
</script>

<div class="message-terminal differences-message" style="color: #5fadaa;">
  <BasicMessage message={message}>

    <svelte:fragment slot="agentNick">[{shortAgentId}]</svelte:fragment>

    <svelte:fragment slot="messagePrefix">
        <i class="bi bi-cpu"></i>|{modelName}|
        <i class="bi bi-file-earmark-text"></i>{message.agent_info?.parameters?.template}
        <i class="bi bi-list-check"></i>{message.agent_info?.parameters?.criteria}
        <i class="bi bi-people"></i> {differencesData.divergences?.reduce((total, div) => total + div.positions.reduce((sum, pos) => sum + pos.experts.length, 0), 0) || 0} experts
    </svelte:fragment>
    
    
    <svelte:fragment slot="messageContent">
      <div class="content-inline">
        {differencesData.conclusion}
        <button class="content-toggle-inline" on:click={toggleContent}
        title={isContentVisible ? "Hide differences" : "Show differences"}>
          {isContentVisible ? '[-]' : '[+]'} differences
        </button>
      
      </div>
    </svelte:fragment>
    
    <svelte:fragment slot="messageExpanded">
      {#if isContentVisible}
          <div class="expanded-content markdown-content" transition:slide={{ duration: 200 }}>
            <!-- Divergences -->
            {#each differencesData.divergences as divergence, index}
                <span class="topic-text">{divergence.topic}</span>
                
                <ol>
                  {#each divergence.positions as position, posIndex}
                      <li class="position-content">
                        <div class="position-text">{position.position}</div>
                        <div class="position-experts">
                          <span class="experts-label">{countExperts(position)} experts:</span> 
                          <span class="experts-list">{formatExpertsList(position.experts)}</span>
                        </div>
                      </li>
                  {/each}
                </ol>
            {/each}
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
          <span class="agent-tag">DIFFERENCES</span> <small>{message.agent_info?.agent_name || 'Differentiator'}</small>
        </span>
        <span class="badge bg-light text-dark">
          {differencesData.divergences?.length || 0} topics
        </span>
      </div>
      
      <div class="card-body p-2">
        <div class="tiny-text">
          <div class="text-truncate">{differencesData.conclusion?.substring(0, 100)}...</div>
          <div><small>
            {differencesData.divergences?.reduce((total, div) => total + div.positions.length, 0) || 0} positions across 
            {differencesData.divergences?.length || 0} topics
          </small></div>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .diff-content {
    padding: 8px 10px;
    background-color: rgba(40, 40, 40, 0.2);
    border-radius: 4px;
    margin-top: 4px;
  }
  
  
  .divergence-item {
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px dotted rgba(80, 80, 80, 0.5);
  }
  
  .divergence-item:last-child {
    border-bottom: none;
    margin-bottom: 0;
  }
  
  .divergence-topic {
    color: #fff;
    font-weight: bold;
    margin-bottom: 6px;
    padding-left: 6px;
  }
  
  .topic-number {
    color: #5fadaa;
    margin-right: 6px;
  }
  
  .position-block {
    background-color: rgba(40, 40, 40, 0.3);
    margin: 6px 0;
    padding: 6px;
    border-radius: 3px;
    border-left: 3px solid rgba(95, 173, 170, 0.5);
  }
  
  .position-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }
  
  .position-marker {
    color: #fff;
    font-weight: bold;
  }
  
  .experts-count {
    color: #aaa;
    font-size: 0.85em;
    cursor: help;
  }
  
  .position-content {
    padding-left: 6px;
    font-size: 0.9em;
  }
  
  .position-experts {
    color: #aaa;
    margin-bottom: 4px;
    font-size: 0.85em;
  }
  
  .experts-label {
    margin-right: 4px;
  }
  
  .experts-list {
    color: #5fadaa;
  }
  
  .position-text {
    color: #e0e0e0;
    line-height: 1.5;
  }
  
  .agent-tag {
    font-weight: bold;
    color: #5fadaa;
  }
</style>
