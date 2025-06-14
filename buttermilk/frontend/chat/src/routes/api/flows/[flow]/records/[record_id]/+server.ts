import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

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
  'osb-002': {
    id: 'osb-002', 
    name: 'Oversight Board Case #002',
    content: 'A post containing satirical content that uses exaggerated stereotypes to make a political point. The content walks the line between legitimate political commentary and potentially harmful generalizations.',
    metadata: {
      created_at: '2024-01-16T14:22:00Z',
      dataset: 'osb',
      word_count: 203,
      char_count: 1142
    }
  },
  'drag-001': {
    id: 'drag-001',
    name: 'Drag Performance Analysis #001',
    content: 'Content celebrating drag culture and its artistic expression, highlighting the creativity and talent in the community. The post discusses the positive impact of drag performances on LGBTQ+ visibility and acceptance.',
    metadata: {
      created_at: '2024-01-17T16:45:00Z',
      dataset: 'drag', 
      word_count: 189,
      char_count: 1067
    }
  },
  'drag-002': {
    id: 'drag-002',
    name: 'Drag Performance Analysis #002', 
    content: 'A post that appears to discuss drag culture but contains subtle coded language that may be interpreted as promoting harmful ideologies when analyzed in context.',
    metadata: {
      created_at: '2024-01-18T09:15:00Z',
      dataset: 'drag',
      word_count: 167,
      char_count: 945
    }
  },
  'tonepolice-001': {
    id: 'tonepolice-001',
    name: 'Tone Policing Example #001',
    content: 'A response to a social justice post that focuses on how the message was delivered rather than its substance, suggesting the author should express their concerns more "civilly" despite addressing legitimate grievances.',
    metadata: {
      created_at: '2024-01-19T11:30:00Z',
      dataset: 'tonepolice',
      word_count: 234,
      char_count: 1334
    }
  },
  'tonepolice-002': {
    id: 'tonepolice-002',
    name: 'Tone Policing Example #002',
    content: 'Content that dismisses valid criticism by focusing on perceived anger or emotion rather than addressing the substantive points raised about systemic issues.',
    metadata: {
      created_at: '2024-01-20T13:45:00Z', 
      dataset: 'tonepolice',
      word_count: 178,
      char_count: 1021
    }
  }
};

export const GET: RequestHandler = async ({ params, fetch, request, url }) => {
  const { record_id, flow } = params;
  const dataset = url.searchParams.get('dataset');
  
  if (!record_id) {
    throw error(400, 'Record ID is required');
  }
  
  if (!flow) {
    throw error(400, 'Flow parameter is required');
  }
  
  // Get backend base URL from environment or use default
  const backendUrl = env.BACKEND_API_URL || 'http://localhost:8080';
  
  try {
    // Create headers for the backend request
    const headers = new Headers();
    headers.append('Accept', 'application/json');
    
    // Forward authorization header if present
    if (request.headers.has('Authorization')) {
      headers.append('Authorization', request.headers.get('Authorization') || '');
    }
    
    // Build backend URL with optional dataset parameter
    const backendUrl_with_params = new URL(`${backendUrl}/api/flows/${encodeURIComponent(flow)}/records/${encodeURIComponent(record_id)}`);
    if (dataset) {
      backendUrl_with_params.searchParams.append('dataset', dataset);
    }
    
    // Try to fetch from backend first using new flow-based endpoint
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