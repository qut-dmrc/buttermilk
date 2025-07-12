import { env } from '$env/dynamic/private';
import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Mock record data based on the API spec for fallback
const MOCK_RECORDS: Record<string, any> = {
  'osb-001': {
    id: 'osb-001',
    name: 'Oversight Board Case #001',
    content: 'Content discussing platform moderation policies and their impact on marginalized communities. This post questions whether certain enforcement actions disproportionately affect specific groups while allowing similar content from other users to remain.',
    metadata: {
      created_at: '2024-01-15T10:30:00Z',
      dataset: 'osb',
      word_count: 156,
      char_count: 892
    }
  },
  'clickhole_trans_bully': {
    id: 'clickhole_trans_bully',
    name: 'Clickhole Trans Bully Content',
    content: 'Sample content for trans-related analysis from Clickhole dataset.',
    metadata: {
      created_at: '2024-01-17T16:45:00Z',
      dataset: 'tja',
      word_count: 189,
      char_count: 1067
    }
  }
};

export const GET: RequestHandler = async ({ params, fetch, request }) => {
  const { record_id, flow, dataset } = params;
  
  if (!record_id) {
    throw error(400, 'Record ID is required');
  }
  
  if (!flow) {
    throw error(400, 'Flow parameter is required');
  }

  if (!dataset) {
    throw error(400, 'Dataset parameter is required');
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
    
    // Build backend URL using dataset-in-path format
    const backendPath = `/api/flows/${encodeURIComponent(flow)}/datasets/${encodeURIComponent(dataset)}/records/${encodeURIComponent(record_id)}`;
    const backendUrl_with_params = new URL(`${backendUrl}${backendPath}`);
    
    // Try to fetch from backend first using flow-based endpoint
    const response = await fetch(backendUrl_with_params.toString(), {
      method: 'GET',
      headers
    });
    
    if (response.ok) {
      const data = await response.json();
      return json(data);
    }
    
    // If backend is unavailable or returns error, fall back to mock data
    console.warn(`Backend unavailable for record ${record_id}, using mock data`);
    
  } catch (backendError) {
    console.warn(`Backend error for record ${record_id}:`, backendError);
  }
  
  // Return mock data if available
  if (MOCK_RECORDS[record_id]) {
    return json(MOCK_RECORDS[record_id]);
  }
  
  // Record not found
  throw error(404, `Record '${record_id}' not found`);
};