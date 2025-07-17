import { browser } from '$app/environment';

// Environment configuration for the application
export const config = {
  // Backend API URL - defaults to localhost for development
  backendApiUrl: browser 
    ? (import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000')
    : 'http://localhost:8000',
  
  // Helper function to build API URLs
  apiUrl: (path: string): string => {
    const baseUrl = browser 
      ? (import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000')
      : 'http://localhost:8000';
    
    // Ensure path starts with /
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    
    return `${baseUrl}${normalizedPath}`;
  }
};

// Export individual values for convenience
export const BACKEND_API_URL = config.backendApiUrl;
