import { env } from '$env/dynamic/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ fetch, request, url }) => {
  // Get the flow parameter from the URL
  const flow = url.searchParams.get('flow');
  
  // If no flow is specified, return an empty array
  if (!flow) {
    return json([]);
  }
  
  // Get backend base URL from environment or use default
  const backendUrl = env.BACKEND_API_URL || 'http://localhost:8000';
  
  try {
    // Create headers for the backend request
    const headers = new Headers();
    headers.append('Accept', 'application/json');
    
    // Forward authorization header if present
    if (request.headers.has('Authorization')) {
      headers.append('Authorization', request.headers.get('Authorization') || '');
    }
    
    // Forward the request to the backend using the new path template format
    const response = await fetch(`${backendUrl}/api/flows/${encodeURIComponent(flow)}/info`, {
      method: 'GET',
      headers
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch records from backend: ${response.statusText}`);
    }
    
    // Return the response from the backend
    const data = await response.json();
    return json(data);
  } catch (error) {
    console.error('Error fetching records from backend:', error);
  }
};
