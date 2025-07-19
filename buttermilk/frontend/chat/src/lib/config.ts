import { browser } from '$app/environment';

// Environment configuration for the application
export const config = {
  // Helper function to build API URLs
  apiUrl: (path: string): string => {
    const baseUrl = browser 
      ? (import.meta.env.BACKEND_API_URL || 'http://localhost:8000')
      : 'http://localhost:8000';
    
    // Ensure path starts with /
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    
    return `${baseUrl}${normalizedPath}`;
  }
};
console.log('Configuration:', config);
// Export individual values for convenience
export const BACKEND_API_URL = browser 
  ? (import.meta.env.BACKEND_API_URL || 'http://localhost:8000')
  : 'http://localhost:8000';
