<script lang="ts">
  import { onMount } from 'svelte';

  let tooltipInstance: any;
  let tooltipElement: HTMLElement;
  let buttonElement: HTMLElement;
  let isVisible = false;

  onMount(() => {
    // Only load Bootstrap in browser environment
    if (typeof window !== 'undefined') {
      import('bootstrap').then(bootstrap => {
        const { Tooltip } = bootstrap;
        tooltipInstance = new Tooltip(tooltipElement, {
          html: true,
          trigger: 'manual',
          template: `
            <div class="tooltip" role="tooltip">
              <div class="tooltip-arrow"></div>
              <div class="tooltip-inner large-tooltip"></div>
            </div>
          `
        });
      }).catch(e => {
        console.warn('Bootstrap not available:', e);
      });
    }

    return () => {
      // Destroy tooltips on component unmount
      if (tooltipInstance) {
        tooltipInstance.dispose();
      }
    };
  });

  function toggleTooltip() {
    isVisible = !isVisible;
    if (isVisible && tooltipInstance) {
      tooltipInstance.show();
    } else if (tooltipInstance) {
      tooltipInstance.hide();
    }
  }
</script>

<div class="test-container">
  <h3>Tooltip Test</h3>
  <p>Testing custom Bootstrap tooltips with Markdown content</p>
  
  <button 
    class="btn btn-primary"
    bind:this={buttonElement}
    on:click={toggleTooltip}
  >
    {isVisible ? 'Hide Content' : 'Show Content'}
  </button>
  
  <div 
    bind:this={tooltipElement}
    data-bs-toggle="tooltip"
    data-bs-placement="bottom"
    data-bs-title="<div class='markdown-content'>
      <h3>Sample Markdown Content</h3>
      <p>This is a <strong>paragraph</strong> with some <em>emphasized</em> text.</p>
      <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
      </ul>
      <blockquote>This is a blockquote</blockquote>
    </div>"
    style="position: absolute; visibility: hidden;"
  ></div>
</div>

<style>
  .test-container {
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 8px;
    margin: 20px 0;
  }
  
  :global(.large-tooltip) {
    max-width: 400px !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    text-align: left !important;
  }
  
  :global(.markdown-content) {
    color: #212529;
    white-space: normal;
  }
  
  :global(.markdown-content h3) {
    font-size: 1.2rem;
    margin-bottom: 10px;
  }
  
  :global(.markdown-content ul) {
    margin-left: 20px;
  }
  
  :global(.markdown-content blockquote) {
    border-left: 3px solid #6c757d;
    padding-left: 10px;
    margin-left: 15px;
    color: #6c757d;
  }
</style>
