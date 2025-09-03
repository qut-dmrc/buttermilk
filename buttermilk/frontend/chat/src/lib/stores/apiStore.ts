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
		timestamp: null
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
				error: errorMessage
			}));
			console.error(`>>> API fetch error for ${url}:`, error);
			return null;
		}
	}

	// Check if cache is still valid
	function isCacheValid() {
		let valid = false;
		const currentState = get(store);
		valid =
			currentState.timestamp !== null &&
			Date.now() - currentState.timestamp < CACHE_TIMEOUT &&
			currentState.data !== null;
		console.log(
			`isCacheValid for ${endpoint}: ${valid} (Timestamp: ${currentState.timestamp}, Now: ${Date.now()})`
		);
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
			timestamp: null
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

// Demo mode store - tracks if backend is unavailable
export const isDemoMode = writable(false);

// 1. Store for initial flow choices
export const initialFlowConfigStore = createApiStore<InitialFlowConfig, InitialFlowConfig>(
	'/api/flows',
	{ flow_choices: [] }
);

export const flowChoices = derived(initialFlowConfigStore, ($config) => ({
	data: $config.data?.flow_choices ?? [],
	loading: $config.loading,
	error: $config.error
}));

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

// 3. Single store for flow-dependent info - will be updated with flow parameter
// Note: endpoint is not used since we manually fetch and update
export const flowInfoStore = createApiStore<FlowInfoResponse | null, FlowInfoResponse>(
	'/api/flows/placeholder/info',
	null,
	(data) => data
);

// 4. Dedicated records store that handles dataset filtering - will be updated with flow parameter
// Note: Use a simple writable store since we manually fetch and update
const recordsStoreInternal = writable<{
	data: RecordItem[];
	loading: boolean;
	error: string | null;
	timestamp: number | null;
}>({
	data: [],
	loading: false,
	error: null,
	timestamp: null
});

export const recordsStore = {
	subscribe: recordsStoreInternal.subscribe,
	_store: recordsStoreInternal,
	reset: function () {
		recordsStoreInternal.set({
			data: [],
			loading: false,
			error: null,
			timestamp: null
		});
	}
};

// Override recordsStore to fetch when flow or dataset changes
export async function refetchRecords() {
	const currentFlow = get(selectedFlow);
	const currentDataset = get(selectedDataset);

	console.log('refetchRecords called with:', { currentFlow, currentDataset });

	// Skip API calls if in demo mode - data is already loaded from sessions
	const currentDemoMode = get(isDemoMode);
	if (currentDemoMode) {
		console.log('Demo mode: Skipping records API call, using cached data');
		return;
	}

	if (currentFlow && currentDataset && currentDataset.trim() !== '') {
		// Always require both flow and dataset - no fallback to flow-only
		const endpoint = `/api/flows/${encodeURIComponent(currentFlow)}/datasets/${encodeURIComponent(currentDataset)}/records`;

		console.log('Fetching records from:', endpoint);

		// Make direct fetch call and update the records store
		try {
			recordsStore._store.update((state) => ({ ...state, loading: true, error: null }));

			const response = await fetch(endpoint);
			if (!response.ok) {
				throw new Error(
					`Error fetching records: ${response.statusText} (Status: ${response.status})`
				);
			}
			const responseData = await response.json();

			console.log('Records fetched successfully:', responseData);

			// Extract records array from response (handle both direct array and {records: [...]} structure)
			const data: RecordItem[] = Array.isArray(responseData)
				? responseData
				: responseData.records || [];

			console.log('Extracted records array:', data);

			// Update recordsStore using its internal writable
			recordsStore._store.update((state) => ({
				...state,
				data: data || [],
				loading: false,
				error: null,
				timestamp: Date.now()
			}));
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
			console.error('Error fetching records:', error);
			recordsStore._store.update((state) => ({
				...state,
				loading: false,
				error: errorMessage,
				data: []
			}));
		}
	} else {
		console.log('Clearing records - missing flow or dataset');
		// Clear records if no flow or no dataset selected
		recordsStore.reset();
	}
}

export const criteriaStore = derived(flowInfoStore, ($info) => ({
	data: $info.data?.criteria ?? [],
	loading: $info.loading,
	error: $info.error
}));


export const datasetsStore = derived(flowInfoStore, ($info) => ({
	data: $info.data?.datasets ?? [],
	loading: $info.loading,
	error: $info.error
}));

// --- Demo Mode Functions ---

// Interface for session data from /api/sessions
interface SessionInfo {
    session_id: string;
    created_at: string;
    last_updated: string;
    flow_status: string;
    parameters: {
        flow?: string;
        dataset?: string;
        record_id?: string;
        criteria?: string;
    };
    message_count: number;
}

// Function to load parameters from available sessions when in demo mode
export async function loadParametersFromSessions() {
    try {
        const response = await fetch('/api/sessions');
        if (!response.ok) {
            throw new Error(`Failed to fetch sessions: ${response.statusText}`);
        }
        
        const data = await response.json() as { sessions?: SessionInfo[] };
        const sessions: SessionInfo[] = data.sessions || [];
        
        // Extract unique values for each parameter type
        const flows = new Set<string>();
        const datasets = new Set<string>();
        const records = new Set<string>();
        const criteriaSet = new Set<string>();
        
        sessions.forEach(session => {
            if (session.parameters?.flow) flows.add(session.parameters.flow);
            if (session.parameters?.dataset) datasets.add(session.parameters.dataset);
            if (session.parameters?.record_id) records.add(session.parameters.record_id);
            if (session.parameters?.criteria) criteriaSet.add(session.parameters.criteria);
        });
        
        // Update the stores with demo data
        initialFlowConfigStore._store.update(state => ({
            ...state,
            data: { flow_choices: Array.from(flows) },
            loading: false,
            error: null,
            timestamp: Date.now()
        }));
        
        // Create mock flow info with available criteria, datasets, and records
        const mockFlowInfo = {
            criteria: Array.from(criteriaSet),
            models: [], // No models needed in demo mode
            datasets: Array.from(datasets),
            record_ids: Array.from(records).map(id => ({ id, name: id }))
        };
        
        flowInfoStore._store.update(state => ({
            ...state,
            data: mockFlowInfo,
            loading: false,
            error: null,
            timestamp: Date.now()
        }));
        
        // Update records store with demo data
        const recordItems = Array.from(records).map(record_id => ({
            record_id,
            name: record_id,
            content: `Demo record: ${record_id}`,
            metadata: {}
        }));
        
        recordsStore._store.update(state => ({
            ...state,
            data: recordItems,
            loading: false,
            error: null,
            timestamp: Date.now()
        }));
        
        console.log('Demo mode: Loaded parameters from sessions', { flows: flows.size, datasets: datasets.size, records: records.size, criteria: criteriaSet.size });
        
    } catch (error) {
        console.error('Failed to load parameters from sessions:', error);
        // Set empty data if loading fails
        initialFlowConfigStore._store.update(state => ({
            ...state,
            data: { flow_choices: [] },
            loading: false,
            error: 'Failed to load demo data',
            timestamp: Date.now()
        }));
    }
}

// Function to find a session that matches the given parameters
export async function findMatchingSession(flow: string, dataset: string, record_id: string, criteria: string): Promise<string | null> {
    try {
        const response = await fetch('/api/sessions');
        if (!response.ok) return null;
        
        const data = await response.json() as { sessions?: SessionInfo[] };
        const sessions: SessionInfo[] = data.sessions || [];
        
        // Find a session with matching parameters
        const matchingSession = sessions.find(session => 
            session.parameters?.flow === flow &&
            session.parameters?.dataset === dataset &&
            session.parameters?.record_id === record_id &&
            session.parameters?.criteria === criteria
        );
        
        return matchingSession?.session_id || null;
    } catch (error) {
        console.error('Failed to find matching session:', error);
        return null;
    }
}

// --- Logic ---

// Fetch initial flow list when app loads
// Track initialization per session to prevent redundant calls
let lastInitializedSession = '';

export async function initializeApp(sessionId?: string) {
    const currentSession = sessionId || 'default';
    
    if (lastInitializedSession === currentSession) {
        console.log(`>>> initializeApp called but already initialized for session ${currentSession}, using cache`);
        // Check if we're already in demo mode, if so don't try API again
        const currentDemoMode = get(isDemoMode);
        if (currentDemoMode) {
            await loadParametersFromSessions();
            return;
        }
        initialFlowConfigStore.fetchWithCache();
        return;
    }
    
    console.log(">>> initializeApp called for session:", currentSession);
    console.log("Initializing app data: fetching flow choices...");
    lastInitializedSession = currentSession;
    
    // Try to fetch from backend first
    try {
        console.log("Attempting to connect to backend...");
        const response = await fetch('/api/flows', { signal: AbortSignal.timeout(5000) });
        
        if (response.ok) {
            console.log("Backend available - using live mode");
            isDemoMode.set(false);
            initialFlowConfigStore.fetchWithCache();
        } else {
            throw new Error(`Backend responded with status: ${response.status}`);
        }
    } catch (error) {
        console.warn("Backend not available, switching to demo mode:", error);
        isDemoMode.set(true);
        await loadParametersFromSessions();
    }
}

// Subscribe to selectedFlow changes to fetch dependent data
selectedFlow.subscribe(async (flowValue) => {
	if (flowValue) {
		console.log(`Selected flow changed to: ${flowValue}. Fetching flow info...`);

		// Skip API calls if in demo mode - data is already loaded from sessions
		const currentDemoMode = get(isDemoMode);
		if (currentDemoMode) {
			console.log('Demo mode: Skipping flow info API call, using cached data');
			return;
		}

		// Fetch flow info using path-based URL
		const flowInfoEndpoint = `/api/flows/${encodeURIComponent(flowValue)}/info`;

		try {
			const response = await fetch(flowInfoEndpoint);
			if (!response.ok) {
				throw new Error(
					`Error fetching flow info: ${response.statusText} (Status: ${response.status})`
				);
			}
			const data: FlowInfoResponse = await response.json();

			// Update flowInfoStore using its internal writable
			flowInfoStore._store.update((state) => ({
				...state,
				data: data || null,
				loading: false,
				error: null,
				timestamp: Date.now()
			}));
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
			flowInfoStore._store.update((state) => ({
				...state,
				loading: false,
				error: errorMessage,
				data: null
			}));
			console.error(`>>> Flow info fetch error for ${flowInfoEndpoint}:`, error);
		}

		// Don't fetch records immediately - wait for dataset selection
		// refetchRecords() will be called when dataset is selected
	} else {
		console.log('Flow selection cleared. Resetting flow info store.');
		flowInfoStore.reset();
		recordsStore.reset();
		// Also reset other selections when flow changes (but not in demo mode)
		const currentDemoMode = get(isDemoMode);
		if (!currentDemoMode) {
			selectedDataset.set('');
			selectedRecord.set('');
			selectedCriteria.set('');
		}
	}
});

// Subscribe to selectedDataset changes to refetch records
selectedDataset.subscribe((datasetValue) => {
	const currentFlow = get(selectedFlow);
	const currentDemoMode = get(isDemoMode);
	
	if (currentFlow && !currentDemoMode) {
		console.log(
			`Selected dataset changed to: ${datasetValue}. Refetching records with dataset filter...`
		);
		refetchRecords();
	} else if (currentDemoMode) {
		console.log('Demo mode: Skipping dataset change record refetch, using cached data');
	}
});

// Derived store for flow selection status
export const hasSelectedFlow = derived(selectedFlow, ($selectedFlow) => $selectedFlow !== '');

// --- Admin Configuration Management ---

// Interface for reload response
interface ConfigReloadResponse {
	success: boolean;
	flows_loaded: string[];
	flows_updated: string[];
	flows_removed: string[];
	errors: string[];
	timestamp: string;
	config_source: string;
}

// Interface for config status response
interface ConfigStatusResponse {
	flows_loaded: string[];
	flow_count: number;
	config_directory: string;
	config_exists: boolean;
	is_gcs_mounted: boolean;
	mount_info: string;
	config_timestamps: Record<string, number>;
	gcs_bucket_env: string;
	timestamp: string;
	error?: string;
}

// Store for configuration reload status
export const configReloadStore = writable<{
	loading: boolean;
	lastResult: ConfigReloadResponse | null;
	error: string | null;
}>({
	loading: false,
	lastResult: null,
	error: null
});

// Store for configuration status
export const configStatusStore = createApiStore<ConfigStatusResponse, ConfigStatusResponse>(
	'/api/admin/config-status',
	null
);

// Function to reload configuration
export async function reloadConfiguration(): Promise<ConfigReloadResponse | null> {
	configReloadStore.update(state => ({ ...state, loading: true, error: null }));
	
	try {
		const response = await fetch('/api/admin/reload-config', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			}
		});
		
		const result: ConfigReloadResponse = await response.json();
		
		configReloadStore.update(state => ({
			...state,
			loading: false,
			lastResult: result,
			error: result.success ? null : result.errors.join(', ')
		}));
		
		// If reload was successful, refresh flow choices and other data
		if (result.success) {
			console.log('Configuration reload successful, refreshing data...');
			initialFlowConfigStore.reset();
			await initialFlowConfigStore.fetch();
			
			// If there's a currently selected flow and it was updated, refresh its info
			const currentFlow = get(selectedFlow);
			if (currentFlow && result.flows_updated.includes(currentFlow)) {
				console.log(`Refreshing info for updated flow: ${currentFlow}`);
				await refetchFlowInfo();
			}
		}
		
		return result;
	} catch (error) {
		const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
		configReloadStore.update(state => ({
			...state,
			loading: false,
			error: errorMessage
		}));
		console.error('Configuration reload failed:', error);
		return null;
	}
}

// Function to get configuration status
export async function getConfigurationStatus(): Promise<ConfigStatusResponse | null> {
	return configStatusStore.fetch();
}
