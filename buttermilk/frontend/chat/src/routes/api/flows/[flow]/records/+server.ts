import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

export const GET: RequestHandler = async ({ fetch, request, params, url }) => {
  const { flow } = params;
  const includeScores = url.searchParams.get('include_scores') === 'true';
  
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
    
    // Build backend URL
    const backendPath = `/api/flows/${encodeURIComponent(flow)}/records`;
    const backendQueryParams = new URLSearchParams();
    if (includeScores) {
      backendQueryParams.append('include_scores', 'true');
    }
    
    const fullBackendUrl = `${backendUrl}${backendPath}${backendQueryParams.toString() ? '?' + backendQueryParams.toString() : ''}`;
    
    // Forward the request to the backend
    const response = await fetch(fullBackendUrl, {
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
    return json({ error: 'Failed to fetch records' }, { status: 500 });
  }
};