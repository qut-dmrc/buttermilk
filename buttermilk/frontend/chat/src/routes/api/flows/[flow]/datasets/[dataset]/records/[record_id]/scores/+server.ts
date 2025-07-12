import { env } from '$env/dynamic/private';
import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Mock score data based on the API spec for fallback
const MOCK_SCORES: Record<string, any> = {
  'clickhole_trans_bully': {
    record_id: 'clickhole_trans_bully',
    off_shelf_results: {
      'GPT-4': {
        correct: true,
        score: 0.92,
        label: 'TOXIC',
        confidence: 0.94,
        model_version: 'gpt-4-0613'
      },
      'Claude-3': {
        correct: true,
        score: 0.89,
        label: 'TOXIC',
        confidence: 0.91,
        model_version: 'claude-3-sonnet'
      },
      'Gemini': {
        correct: true,
        score: 0.85,
        label: 'TOXIC',
        confidence: 0.88,
        model_version: 'gemini-pro'
      },
      'LLaMA-2': {
        correct: true,
        score: 0.93,
        label: 'TOXIC',
        confidence: 0.95,
        model_version: 'llama-2-70b'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.92,
        model: 'gpt-4-0613',
        template: 'trans_judge_v1',
        criteria: 'trans_advocacy'
      },
      'Judge-Claude': {
        step: 'judge',
        score: 0.89,
        model: 'claude-3-sonnet',
        template: 'trans_judge_v1',
        criteria: 'trans_advocacy'
      },
      'Synth-GPT4': {
        step: 'synth',
        score: 0.91,
        model: 'gpt-4-0613',
        template: 'synthesis_v1',
        criteria: 'trans_advocacy'
      }
    },
    summary: {
      off_shelf_accuracy: 1.0,
      custom_average_score: 0.907,
      total_evaluations: 7,
      agreement_rate: 0.98
    }
  },
  'osb-001': {
    record_id: 'osb-001',
    off_shelf_results: {
      'GPT-4': {
        correct: true,
        score: 0.85,
        label: 'TOXIC',
        confidence: 0.92,
        model_version: 'gpt-4-0613'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.88,
        model: 'gpt-4-0613',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      }
    },
    summary: {
      off_shelf_accuracy: 1.0,
      custom_average_score: 0.88,
      total_evaluations: 2,
      agreement_rate: 1.0
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
    const backendPath = `/api/flows/${encodeURIComponent(flow)}/datasets/${encodeURIComponent(dataset)}/records/${encodeURIComponent(record_id)}/scores`;
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
    console.warn(`Backend unavailable for scores ${record_id}, using mock data`);
    
  } catch (backendError) {
    console.warn(`Backend error for scores ${record_id}:`, backendError);
  }
  
  // Return mock data if available
  if (MOCK_SCORES[record_id]) {
    return json(MOCK_SCORES[record_id]);
  }
  
  // Record not found
  throw error(404, `Scores for record '${record_id}' not found`);
};