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
  record_ids?: { id: string; name: string }[];
}

// For a single record item (adjust as needed)
interface RecordItem {
  id: string;
  name: string;
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
    reset
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

// 3. Single store for flow-dependent info
export const flowInfoStore = createApiStore<FlowInfoResponse | null, FlowInfoResponse>(
    '/api/flowinfo',
    null,
    (data) => data
);

// 4. Derived stores for specific data points from flowInfoStore
export const recordsStore = derived(
    flowInfoStore,
    ($info) => {
        const recordData = $info.data?.record_ids ?? [];
        
        return {
            data: recordData,
            loading: $info.loading,
            error: $info.error
        };
    }
);

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

// --- Logic ---

// Fetch initial flow list when app loads
export function initializeApp() {
    console.log(">>> initializeApp called");
    console.log("Initializing app data: fetching flow choices...");
    initialFlowConfigStore.fetchWithCache();
}

// Subscribe to selectedFlow changes to fetch dependent data
selectedFlow.subscribe((flowValue) => {
  if (flowValue) {
    console.log(`Selected flow changed to: ${flowValue}. Fetching flow info...`);
    const params = { flow: flowValue };
    flowInfoStore.fetch(params);
  } else {
    console.log("Flow selection cleared. Resetting flow info store.");
    flowInfoStore.reset();
    // Also reset other selections when flow changes
    selectedRecord.set('');
    selectedCriteria.set('');
    selectedModel.set('');
  }
});

// Derived store for flow selection status
export const hasSelectedFlow = derived(
  selectedFlow,
  ($selectedFlow) => $selectedFlow !== ''
);
