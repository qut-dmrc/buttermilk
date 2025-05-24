<script lang="ts">
  // Import for rendering markdown
  import { onMount } from 'svelte';
  
  // Get the data from the server
  export let data;
  
  let htmlContent = '';
  
  onMount(() => {
    // Check if marked library is available
    if (typeof window !== 'undefined' && 'marked' in window) {
      // Use marked to render markdown if available
      const markedLib = (window as any).marked;
      htmlContent = markedLib.parse(data.content);
    } else {
      // Simple fallback markdown renderer
      htmlContent = data.content
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
        .replace(/^##### (.*$)/gm, '<h5>$1</h5>')
        .replace(/^###### (.*$)/gm, '<h6>$1</h6>')
        .replace(/^\> (.*$)/gm, '<blockquote>$1</blockquote>')
        .replace(/\*\*(.*)\*\*/gm, '<strong>$1</strong>')
        .replace(/\*(.*)\*/gm, '<em>$1</em>')
        .replace(/\`\`\`([\s\S]*?)\`\`\`/gm, '<pre><code>$1</code></pre>')
        .replace(/\`(.*?)\`/gm, '<code>$1</code>')
        .replace(/!\[(.*?)\]\((.*?)\)/gm, '<img alt="$1" src="$2">')
        .replace(/\[(.*?)\]\((.*?)\)/gm, '<a href="$2">$1</a>')
        .replace(/^\s*\n\* (.*)/gm, '<ul>\n<li>$1</li>\n</ul>')
        .replace(/^\s*\n\d\. (.*)/gm, '<ol>\n<li>$1</li>\n</ol>')
        .replace(/^\s*\n- (.*)/gm, '<ul>\n<li>$1</li>\n</ul>')
        .replace(/\n{2,}/g, '</p><p>')
        .replace(/\n/g, '<br>');
    }
  });
</script>

<svelte:head>
  <title>{data.title} | Terminal Console Documentation</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</svelte:head>

<div class="container py-4">
  <a href="/" class="back-link btn btn-sm btn-secondary mb-4">
    &larr; Back to Terminal Console
  </a>
  
  <div class="markdown-content">
    <h1>{data.title}</h1>
    
    <!-- Display the rendered markdown -->
    {@html htmlContent}
  </div>
</div>
