/**
 * Centralized backend connectivity utilities
 *
 * Provides reusable functions for checking backend availability
 * with caching to avoid repeated network calls.
 */

interface BackendHealthCache {
	available: boolean;
	lastCheck: number;
	error?: string;
}

// Cache backend health status for 30 seconds to avoid excessive health checks
const CACHE_DURATION = 30 * 1000; // 30 seconds
const HEALTH_CHECK_TIMEOUT = 3000; // 3 seconds

let healthCache: BackendHealthCache | null = null;

/**
 * Check if backend is available by testing the session endpoint
 * This is the most reliable health check as it verifies core functionality
 *
 * @param useCache - Whether to use cached result if available (default: true)
 * @param fetchFn - Custom fetch function (for SvelteKit server-side use event.fetch)
 * @returns Promise resolving to true if backend is available
 */
export async function checkBackendHealth(useCache: boolean = true, fetchFn?: typeof fetch): Promise<boolean> {
	// Return cached result if valid and requested
	if (useCache && healthCache && (Date.now() - healthCache.lastCheck) < CACHE_DURATION) {
		return healthCache.available;
	}

	try {
		const useFetch = fetchFn || fetch;
		const response = await useFetch('/api/session', {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
			},
			signal: AbortSignal.timeout(HEALTH_CHECK_TIMEOUT)
		});

		const available = response.ok;

		// Update cache
		healthCache = {
			available,
			lastCheck: Date.now(),
			error: available ? undefined : `HTTP ${response.status}: ${response.statusText}`
		};

		return available;
	} catch (error) {
		const errorMsg = error instanceof Error ? error.message : 'Unknown error';

		// Update cache with error
		healthCache = {
			available: false,
			lastCheck: Date.now(),
			error: errorMsg
		};

		return false;
	}
}

/**
 * Synchronously check if backend is available using cached data
 * Returns null if no cached data available - use checkBackendHealth() first
 *
 * @returns boolean if cached data available, null otherwise
 */
export function isBackendAvailable(): boolean | null {
	if (!healthCache || (Date.now() - healthCache.lastCheck) >= CACHE_DURATION) {
		return null; // No valid cached data
	}
	return healthCache.available;
}

/**
 * Get the last known backend error message
 * Useful for logging purposes
 *
 * @returns Error message if backend is unavailable, null if available or no cache
 */
export function getBackendError(): string | null {
	if (!healthCache || healthCache.available) {
		return null;
	}
	return healthCache.error || 'Backend unavailable';
}

/**
 * Clear the backend health cache
 * Forces next health check to hit the network
 */
export function clearBackendHealthCache(): void {
	healthCache = null;
}

/**
 * Log backend connectivity status in a consistent, concise format
 *
 * @param context - Context string for the log message (e.g., "WebSocket connection", "API call")
 * @param available - Whether backend is available
 */
export function logBackendStatus(context: string, available: boolean): void {
	if (available) {
		console.debug(`${context}: Backend available`);
	} else {
		const error = getBackendError();
		console.log(`${context}: Backend unavailable${error ? ` (${error})` : ''}`);
		console.warn(`${context}: Backend unavailable${error ? ` (${error})` : ''}`);
	}
}

/**
 * Get the configured sessions directory from environment variable
 * Falls back to default path if not configured
 *
 * @returns Configured sessions directory path
 */
export function getSessionsDir(): string {
	// Use environment variable or fallback to default
	// Use process.env for server-side (Node.js) or import.meta.env for client-side (Vite)
	const sessionsDir = (typeof process !== 'undefined' ? process.env.SESSIONS_DIR : import.meta.env.SESSIONS_DIR) || 'data/sessions';
	console.debug(`Using sessions directory: ${sessionsDir}`);
	return sessionsDir;
}
