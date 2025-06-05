<script lang="ts">
  export let recordData: {
    id: string;
    name: string;
    content: string;
  };

  function formatContent(content: string): string {
    // Basic content formatting - could be enhanced with markdown support
    return content.replace(/\n/g, '<br>');
  }
</script>

<div class="record-display">
  <div class="content-header">
    <span class="content-label">CONTENT:</span>
    <span class="content-meta">ID: {recordData.id}</span>
  </div>
  
  <div class="content-body">
    <div class="content-text">
      {@html formatContent(recordData.content)}
    </div>
  </div>
  
  <div class="content-footer">
    <div class="content-stats">
      <span class="stat-item">
        <span class="stat-label">LENGTH:</span>
        <span class="stat-value">{recordData.content.length} chars</span>
      </span>
      <span class="stat-item">
        <span class="stat-label">WORDS:</span>
        <span class="stat-value">{recordData.content.split(/\s+/).length}</span>
      </span>
    </div>
  </div>
</div>

<style>
  .record-display {
    background-color: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    overflow: hidden;
  }

  .content-header {
    background-color: rgba(0, 255, 0, 0.1);
    border-bottom: 1px solid rgba(0, 255, 0, 0.3);
    padding: 0.75rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .content-label {
    color: #00ff00;
    font-weight: bold;
    font-size: 0.9rem;
  }

  .content-meta {
    color: #888;
    font-size: 0.8rem;
  }

  .content-body {
    padding: 1.5rem;
  }

  .content-text {
    color: #e0e0e0;
    line-height: 1.6;
    font-size: 1rem;
    background-color: rgba(255, 255, 255, 0.02);
    padding: 1rem;
    border-left: 3px solid rgba(0, 255, 255, 0.3);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .content-footer {
    background-color: rgba(0, 0, 0, 0.2);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding: 0.5rem 1rem;
  }

  .content-stats {
    display: flex;
    gap: 2rem;
    font-size: 0.8rem;
  }

  .stat-item {
    display: flex;
    gap: 0.5rem;
  }

  .stat-label {
    color: #00ffff;
    font-weight: bold;
  }

  .stat-value {
    color: #ccc;
  }
</style>