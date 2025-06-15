import { writable, derived, get } from 'svelte/store';

// --- Interfaces ---
// For the initial /api/flows response
interface InitialFlowConfig {
  flow_choices: string[];
}

// For the /api/flowinfo response 
interface FlowInfoResponse {
  criteria?: string[];
  models?: string[];
  datasets?: string[];
  record_ids?: { id: string; name: string }[];
}

// For a single record item (matching backend Record model)
interface RecordItem {
  record_id: string;
  name?: string;
  content?: string;
  metadata?: any;
}

// --- Generic API Store Creator ---
function createApiStore<T, R>(
  endpoint: string,
  initialValue: T | null = null,
  transformResponse: (data: R) => T = (data: any) => data as T
) {
  const store = writable<{
    data: T | null;
    loading: boolean;
    error: string | null;
    timestamp: number | null;
  }>({
    data: initialValue,
    loading: false,
    error: null,
    timestamp: null,
  });

  const CACHE_TIMEOUT = 5 * 60 * 1000; // 5 minutes

  // Function to fetch data
  async function fetch(params?: Record<string, string>) {

    store.update((state) => ({ ...state, loading: true, error: null }));
    let url = endpoint;
    if (params && Object.keys(params).length > 0) {
      const queryParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value) queryParams.append(key, value);
      });
      url = `${endpoint}?${queryParams.toString()}`;
    }

    try {
      const response = await window.fetch(url);

      if (!response.ok) {
        throw new Error(`Error fetching data: ${response.statusText} (Status: ${response.status})`);
      }
      const responseData: R = await response.json();
      const data = transformResponse(responseData);
      store.update((state) => ({
        ...state,
        data,
        loading: false,
        error: null,
        timestamp: Date.now()
      }));
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      store.update((state) => ({
        ...state,
        loading: false,
        error: errorMessage,
      }));
      console.error(`>>> API fetch error for ${url}:`, error);
      return null;
    }
  }

  // Check if cache is still valid
  function isCacheValid() {
    let valid = false;
    const currentState = get(store);
    valid = currentState.timestamp !== null &&
            (Date.now() - currentState.timestamp < CACHE_TIMEOUT) &&
            currentState.data !== null;
    console.log(`isCacheValid for ${endpoint}: ${valid} (Timestamp: ${currentState.timestamp}, Now: ${Date.now()})`);
    return valid;
  }

  // Function to fetch with caching
  function fetchWithCache(params?: Record<string, string>, forceFresh = false) {
    if (!forceFresh && isCacheValid()) {
      console.log(`Using cached data for ${endpoint}`);
      return;
    }
    console.log(`Fetching fresh data for ${endpoint} with params:`, params);
    return fetch(params);
  }

  function reset() {
    console.log(`Resetting store for endpoint: ${endpoint}`);
    store.set({
      data: initialValue,
      loading: false,
      error: null,
      timestamp: null,
    });
  }

  const { subscribe } = store;

  return {
    subscribe,
    fetch,
    fetchWithCache,
    reset,
    // Expose internal store for manual updates
    _store: store
  };
}

// --- Stores ---

export const flowRunning = writable(false);
// 1. Store for initial flow choices
export const initialFlowConfigStore = createApiStore<InitialFlowConfig, InitialFlowConfig>(
    '/api/flows',
    { flow_choices: [] }
  );

export const flowChoices = derived(
  initialFlowConfigStore,
  ($config) => ({
    data: $config.data?.flow_choices ?? [],
    loading: $config.loading,
    error: $config.error
  })
);

// 2. Stores for the currently selected API parameters
export const selectedFlow = writable<string>('');
export const selectedDataset = writable<string>('');

// Create selectedRecord with a custom set method to log changes
const createSelectedRecordStore = () => {
  const { subscribe, set: originalSet, update } = writable<string>('');
  
  const set = (value: string) => {
    originalSet(value);
  };
  
  return {
    subscribe,
    set,
    update
  };
};

export const selectedRecord = createSelectedRecordStore();
export const selectedCriteria = writable<string>('');
export const selectedModel = writable<string>('');

// 3. Single store for flow-dependent info - will be updated with flow parameter
// Note: endpoint is not used since we manually fetch and update
export const flowInfoStore = createApiStore<FlowInfoResponse | null, FlowInfoResponse>(
    '/api/flows/placeholder/info',
    null,
    (data) => data
);

// 4. Dedicated records store that handles dataset filtering - will be updated with flow parameter
// Note: endpoint is not used since we manually fetch and update
export const recordsStore = createApiStore<RecordItem[], RecordItem[]>(
    '/api/flows/placeholder/records',
    []
);

// Override recordsStore to fetch when flow or dataset changes
async function refetchRecords() {
  const currentFlow = get(selectedFlow);
  const currentDataset = get(selectedDataset);
  
  if (currentFlow && currentDataset && currentDataset.trim() !== '') {
    // Always require both flow and dataset - no fallback to flow-only
    const endpoint = `/api/flows/${encodeURIComponent(currentFlow)}/datasets/${encodeURIComponent(currentDataset)}/records`;
    
    // Make direct fetch call and update the records store
    try {
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error(`Error fetching records: ${response.statusText} (Status: ${response.status})`);
      }
      const data: RecordItem[] = await response.json();
      
      // Update recordsStore using its internal writable
      recordsStore._store.update(state => ({
        ...state,
        data: data || [],
        loading: false,
        error: null,
        timestamp: Date.now()
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      recordsStore._store.update(state => ({
        ...state,
        loading: false,
        error: errorMessage,
        data: []
      }));
      console.error(`>>> Records fetch error for ${endpoint}:`, error);
    }
  } else {
    // Clear records if no flow or no dataset selected
    recordsStore.reset();
  }
}

export const criteriaStore = derived(
    flowInfoStore,
    ($info) => ({
        data: $info.data?.criteria ?? [],
        loading: $info.loading,
        error: $info.error
    })
);

export const modelStore = derived(
    flowInfoStore,
    ($info) => ({
        data: $info.data?.models ?? [], // Corrected 'model' to 'models'
        loading: $info.loading,
        error: $info.error
    })
);

export const datasetsStore = derived(
    flowInfoStore,
    ($info) => ({
        data: $info.data?.datasets ?? [],
        loading: $info.loading,
        error: $info.error
    })
);

// --- Logic ---

// Fetch initial flow list when app loads
export function initializeApp() {
    console.log(">>> initializeApp called");
    console.log("Initializing app data: fetching flow choices...");
    initialFlowConfigStore.fetchWithCache();
}

// Subscribe to selectedFlow changes to fetch dependent data
selectedFlow.subscribe(async (flowValue) => {
  if (flowValue) {
    console.log(`Selected flow changed to: ${flowValue}. Fetching flow info...`);
    
    // Fetch flow info using path-based URL
    const flowInfoEndpoint = `/api/flows/${encodeURIComponent(flowValue)}/info`;
    
    try {
      const response = await fetch(flowInfoEndpoint);
      if (!response.ok) {
        throw new Error(`Error fetching flow info: ${response.statusText} (Status: ${response.status})`);
      }
      const data: FlowInfoResponse = await response.json();
      
      // Update flowInfoStore using its internal writable
      flowInfoStore._store.update(state => ({
        ...state,
        data: data || null,
        loading: false,
        error: null,
        timestamp: Date.now()
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      flowInfoStore._store.update(state => ({
        ...state,
        loading: false,
        error: errorMessage,
        data: null
      }));
      console.error(`>>> Flow info fetch error for ${flowInfoEndpoint}:`, error);
    }
    
    refetchRecords();
  } else {
    console.log("Flow selection cleared. Resetting flow info store.");
    flowInfoStore.reset();
    recordsStore.reset();
    // Also reset other selections when flow changes
    selectedDataset.set('');
    selectedRecord.set('');
    selectedCriteria.set('');
    selectedModel.set('');
  }
});

// Subscribe to selectedDataset changes to refetch records
selectedDataset.subscribe((datasetValue) => {
  const currentFlow = get(selectedFlow);
  if (currentFlow) {
    console.log(`Selected dataset changed to: ${datasetValue}. Refetching records with dataset filter...`);
    refetchRecords();
  }
});

// Derived store for flow selection status
export const hasSelectedFlow = derived(
  selectedFlow,
  ($selectedFlow) => $selectedFlow !== ''
);
