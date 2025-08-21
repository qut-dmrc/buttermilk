import { env } from '$env/dynamic/private';
// Environment configuration for the application
export const config = {
	// Helper function to build API URLs
	apiUrl: (path: string): string => {
		const baseUrl = env.BACKEND_API_URL;

		// Ensure path starts with /
		const normalizedPath = path.startsWith('/') ? path : `/${path}`;

		return `${baseUrl}${normalizedPath}`;
	}
};
console.log('Configuration:', config);
// Export individual values for convenience
export const BACKEND_API_URL = env.BACKEND_API_URL;
