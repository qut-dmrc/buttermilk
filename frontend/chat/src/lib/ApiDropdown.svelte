<script lang="ts">
  import { onMount } from 'svelte';
  import type { Readable } from 'svelte/store';

  // Props - Type can expect the full structure again
  export let store: {
    subscribe: Readable<{
      data: any;
      loading: boolean;
      error: string | null;
      timestamp?: number | null; // Keep timestamp optional here
    }>['subscribe'];
    // Fetch methods are still optional because derived stores won't have them
    fetch?: (params?: Record<string, string>) => Promise<any>;
    fetchWithCache?: (params?: Record<string, string>, forceFresh?: boolean) => Promise<any> | undefined;
    reset?: () => void;
  };
  export let label: string = '';
  export let placeholder: string = 'Select an option';
  export let value: string = '';
  
  // Debug value changes
  $: {
    if (label === 'Select Record' && value) {
      console.log(`Record selected in dropdown: value=${value}, type=${typeof value}`);
    }
  }
  export let disabled: boolean = false;
  export let required: boolean = false;
  export let fetchParams: Record<string, string> = {};
  export let valueProperty: string = '';
  export let labelProperty: string = '';
  export let isPlainArray: boolean = false;

  // Component state
  let loading = false;
  let error = '';
  let data: any[] = [];
  let mounted = false;
    // Determine if the passed store is a full API store or a simple data store
    let isApiStore = typeof store.fetch === 'function'; // Keep this check

  // Subscribe to store changes - Simplified back
  const unsubscribe = store.subscribe(storeValue => {
    // Always expect the object structure now
    loading = storeValue?.loading ?? false;
    error = storeValue?.error || '';
    const rawData = storeValue?.data;
    
    // Debug logging for record dropdown
    if (label === 'Select Record') {
      console.log(`Record dropdown data:`, rawData);
    }
    
    // Process rawData based on isPlainArray (same logic as before)
    if (isPlainArray && Array.isArray(rawData)) {
        data = rawData;
    } else if (Array.isArray(rawData)) {
        data = rawData;
    } else {
        data = [];
    }
  });

  // onMount logic remains the same (checking if fetchWithCache exists)
  onMount(() => {
    mounted = true;
    console.log(`ApiDropdown mounted for label: "${label}"`);
    console.log(`  Store object passed:`, store);
    const hasFetchWithCache = typeof store.fetchWithCache === 'function';
    console.log(`  Is this an API store (has fetch)?`, isApiStore);
    console.log(`  Does store have fetchWithCache method?`, hasFetchWithCache);

    // Fetch data on mount ONLY if it's a full API store
    if (isApiStore && hasFetchWithCache) {
        console.log(`  Attempting to call fetchWithCache for "${label}"`);
        try {
            store.fetchWithCache!(fetchParams);
        } catch (e) {
            console.error(`  Error calling fetchWithCache for "${label}":`, e);
            error = `Error during initial fetch: ${e instanceof Error ? e.message : String(e)}`;
        }
    } else {
        console.log(`  Skipping initial fetchWithCache for "${label}" (not an API store or method missing)`);
    }

    return () => {
      unsubscribe();
    };
  });

  // handleRefresh and canRefresh remain the same (checking if fetch exists)
  function handleRefresh() {
    if (isApiStore && typeof store.fetch === 'function') {
        store.fetch(fetchParams);
    }
  }
  $: canRefresh = isApiStore && !!store.fetch;

  // Get the label to display for each option
  function getOptionLabel(item: any): string {
    // Explicitly check isPlainArray first
    if (isPlainArray) {
      // For plain arrays, the item itself is the label
      return String(item ?? ''); // Ensure it's a string, handle null/undefined
    } else if (labelProperty && item && typeof item === 'object') {
      // For object arrays, use the labelProperty
      return String(item[labelProperty] ?? ''); // Ensure it's a string, handle missing prop
    } else {
      // Fallback for unexpected types or missing labelProperty
      return String(item ?? '');
    }
  }

  // Get the value for each option
  function getOptionValue(item: any): string {
    let result = '';
    
    // Explicitly check isPlainArray first
    if (isPlainArray) {
      // For plain arrays, the item itself is the value
      result = String(item ?? ''); // Ensure it's a string, handle null/undefined
    } else if (valueProperty && item && typeof item === 'object') {
      // For object arrays, use the valueProperty
      const propValue = item[valueProperty];
      
      // Debug for record dropdown
      if (label === 'Select Record') {
        console.log(`Record option processing: item=${JSON.stringify(item)}, valueProperty=${valueProperty}, propValue=${propValue}`);
      }
      
      // Make sure we have a non-empty value
      if (propValue === undefined || propValue === null || propValue === '') {
        // If no ID is found but we have a name, use that instead
        if (labelProperty && item[labelProperty]) {
          result = String(item[labelProperty]);
          console.log(`Using name as fallback ID: '${result}'`);
        } else {
          // Last resort - use the whole item stringified
          result = JSON.stringify(item);
          console.log(`Using stringified item as fallback: '${result}'`);
        }
      } else {
        // Normal case - use the ID property
        result = String(propValue);
      }
    } else {
      // Fallback for unexpected types or missing valueProperty
      result = String(item ?? '');
    }
    
    if (label === 'Select Record') {
      console.log(`Final option value: '${result}'`);
    }
    
    return result;
  }
</script>

<div class="api-dropdown mb-3">
  {#if label}
    <label for="api-select-{label.toLowerCase().replace(/\s+/g, '-')}" class="form-label">
      {label}
      {#if loading}
        <span class="spinner-border spinner-border-sm ms-2" role="status" aria-hidden="true"></span>
      {/if}
    </label>
  {/if}

  <div class="input-group">
    <select
      id="api-select-{label ? label.toLowerCase().replace(/\s+/g, '-') : 'dropdown'}"
      class="form-select {error ? 'is-invalid' : ''}"
      bind:value
      {disabled}
      {required}
      aria-label={label || 'Select option'}
      on:change={(e) => {
        if (label === 'Select Record') {
          const select = e.target as HTMLSelectElement;
          console.log(`Select element change event:`, e);
          console.log(`Selected value: ${select?.value}, Selected index: ${select?.selectedIndex}`);
          console.log(`Selected option:`, select?.options[select?.selectedIndex]);
          
          // Force update the value if needed
          if (select?.value && value !== select.value) {
            console.log(`Value mismatch! Dropdown: ${select.value}, Bound: ${value}`);
            // Force update the value (should happen automatically with binding)
            value = select.value;
          }
        }
        // Forward the change event to parent components
      }}
    >
      <option value="" disabled selected>{placeholder}</option>
      {#if data && data.length > 0}
        {#each data as item, index (index + '-' + getOptionValue(item))}
          <option value={getOptionValue(item)}>
            {getOptionLabel(item)}
          </option>
        {/each}
      {/if}
    </select>

    {#if canRefresh}
      <button
        class="btn btn-outline-secondary"
        type="button"
        on:click={handleRefresh}
        disabled={loading || disabled}
        title="Refresh options"
      >
        {#if loading}
          <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
          <span class="visually-hidden">Loading...</span>
        {:else}
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-clockwise" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2z"/>
            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466"/>
          </svg>
        {/if}
      </button>
    {/if}
  </div>

  {#if error}
    <div class="invalid-feedback d-block">
      Error loading options: {error}
    </div>
  {/if}

  {#if !loading && !error && data && data.length === 0 && mounted}
    <small class="text-muted d-block mt-1">No options available</small>
  {/if}
</div>

<style>
  .api-dropdown {
    width: 100%;
  }
  .invalid-feedback.d-block {
      display: block !important;
  }
</style>
