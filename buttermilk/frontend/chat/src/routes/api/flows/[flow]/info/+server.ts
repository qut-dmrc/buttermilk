import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

export const GET: RequestHandler = async ({ fetch, request, params }) => {
  const { flow } = params;
  
  // If no flow is specified, return error
  if (!flow) {
    return json({ error: 'Flow parameter is required' }, { status: 400 });
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
    
    // Forward the request to the backend using path template format
    const response = await fetch(`${backendUrl}/api/flows/${encodeURIComponent(flow)}/info`, {
      method: 'GET',
      headers
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch flow info from backend: ${response.statusText}`);
    }
    
    // Return the response from the backend
    const data = await response.json();
    return json(data);
  } catch (error) {
    console.error('Error fetching flow info from backend:', error);
    return json({ error: 'Failed to fetch flow info' }, { status: 500 });
  }
};