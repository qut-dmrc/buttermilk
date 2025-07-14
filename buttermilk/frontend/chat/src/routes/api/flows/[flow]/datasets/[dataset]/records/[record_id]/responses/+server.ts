import { env } from '$env/dynamic/private';
import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Mock responses data based on the API spec for fallback
const MOCK_RESPONSES: Record<string, any> = {
  'clickhole_trans_bully': {
    record_id: 'clickhole_trans_bully',
    responses: [
      {
        agent: 'Judge-GPT4',
        type: 'judge',
        model: 'gpt-4-0613',
        content: 'This content contains problematic language targeting transgender individuals.',
        score: 0.92,
        reasoning: 'The content uses derogatory terms and promotes harmful stereotypes about transgender people. Despite being from a satirical source, the language employed can contribute to discrimination and harassment.',
        criteria_used: 'trans_advocacy',
        template: 'trans_judge_v1',
        timestamp: '2024-01-15T10:30:00Z',
        confidence: 0.94,
        prediction: true
      },
      {
        agent: 'Judge-Claude',
        type: 'judge',
        model: 'claude-3-sonnet',
        content: 'Content exhibits clear signs of anti-transgender bias and harmful rhetoric.',
        score: 0.89,
        reasoning: 'Analysis reveals multiple markers of transphobic content including misgendering, use of slurs, and promotion of discriminatory attitudes. The satirical context does not mitigate the harmful impact.',
        criteria_used: 'trans_advocacy',
        template: 'trans_judge_v1',
        timestamp: '2024-01-15T10:31:00Z',
        confidence: 0.91,
        prediction: true
      },
      {
        agent: 'Synth-GPT4',
        type: 'synth',
        model: 'gpt-4-0613',
        content: 'Synthesis confirms strong consensus on harmful content classification.',
        score: 0.91,
        reasoning: 'Multiple evaluation perspectives align on the harmful nature of this content. The consistency across different models and criteria indicates high confidence in the violation assessment.',
        criteria_used: 'trans_advocacy',
        template: 'synthesis_v1',
        timestamp: '2024-01-15T10:32:00Z',
        confidence: 0.93,
        prediction: true
      }
    ]
  },
  'osb-001': {
    record_id: 'osb-001',
    responses: [
      {
        agent: 'Judge-GPT4',
        type: 'judge',
        model: 'gpt-4-0613',
        content: 'This content violates our community guidelines regarding hate speech targeting specific groups.',
        score: 0.88,
        reasoning: 'The language used contains derogatory terms and promotes harmful stereotypes against the mentioned communities. While the post frames itself as questioning enforcement policies, the underlying rhetoric employs coded language that targets marginalized groups.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-15T10:30:00Z',
        confidence: 0.92,
        prediction: true
      }
    ]
  }
};

export const GET: RequestHandler = async ({ params, fetch, request, url }) => {
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
  
  // Get query parameters
  const include_reasoning = url.searchParams.get('include_reasoning') !== 'false'; // Default true
  
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
    const backendPath = `/api/flows/${encodeURIComponent(flow)}/datasets/${encodeURIComponent(dataset)}/records/${encodeURIComponent(record_id)}/responses`;
    
    // Add query params for include_reasoning
    const backendQueryParams = new URLSearchParams();
    if (!include_reasoning) {
      backendQueryParams.append('include_reasoning', 'false');
    }
    const backendUrlWithParams = `${backendUrl}${backendPath}${backendQueryParams.toString() ? '?' + backendQueryParams.toString() : ''}`;
    
    // Try to fetch from backend first
    const response = await fetch(backendUrlWithParams, {
      method: 'GET',
      headers
    });
    
    if (response.ok) {
      const data = await response.json();
      return json(data);
    }
    
    // If backend is unavailable or returns error, fall back to mock data
    console.warn(`Backend unavailable for responses ${record_id}, using mock data`);
    
  } catch (backendError) {
    console.warn(`Backend error for responses ${record_id}:`, backendError);
  }
  
  // Return mock data if available
  if (MOCK_RESPONSES[record_id]) {
    let data = MOCK_RESPONSES[record_id];
    
    // Filter out reasoning if not requested
    if (!include_reasoning) {
      data = {
        ...data,
        responses: data.responses.map((response: any) => {
          const { reasoning, ...responseWithoutReasoning } = response;
          return responseWithoutReasoning;
        })
      };
    }
    
    return json(data);
  }
  
  // Record not found
  throw error(404, `Responses for record '${record_id}' not found`);
};