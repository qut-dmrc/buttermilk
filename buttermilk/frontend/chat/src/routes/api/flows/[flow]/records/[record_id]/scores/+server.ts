import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

// Mock score data based on the API spec for fallback
const MOCK_SCORES: Record<string, any> = {
  'osb-001': {
    record_id: 'osb-001',
    off_shelf_results: {
      'GPT-4': {
        correct: true,
        score: 0.85,
        label: 'TOXIC',
        confidence: 0.92,
        model_version: 'gpt-4-0613'
      },
      'Claude-3': {
        correct: false,
        score: 0.42,
        label: 'SAFE',
        confidence: 0.78,
        model_version: 'claude-3-sonnet'
      },
      'Gemini': {
        correct: true,
        score: 0.78,
        label: 'TOXIC',
        confidence: 0.85,
        model_version: 'gemini-pro'
      },
      'LLaMA-2': {
        correct: true,
        score: 0.91,
        label: 'TOXIC',
        confidence: 0.88,
        model_version: 'llama-2-70b'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.88,
        model: 'gpt-4-0613',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Judge-Claude': {
        step: 'judge',
        score: 0.45,
        model: 'claude-3-sonnet',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Synth-GPT4': {
        step: 'synth',
        score: 0.82,
        model: 'gpt-4-0613',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      },
      'Synth-Claude': {
        step: 'synth',
        score: 0.39,
        model: 'claude-3-sonnet',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      }
    },
    summary: {
      off_shelf_accuracy: 0.75,
      custom_average_score: 0.635,
      total_evaluations: 8,
      agreement_rate: 0.62
    }
  },
  'osb-002': {
    record_id: 'osb-002',
    off_shelf_results: {
      'GPT-4': {
        correct: false,
        score: 0.35,
        label: 'SAFE',
        confidence: 0.89,
        model_version: 'gpt-4-0613'
      },
      'Claude-3': {
        correct: true,
        score: 0.72,
        label: 'TOXIC',
        confidence: 0.81,
        model_version: 'claude-3-sonnet'
      },
      'Gemini': {
        correct: false,
        score: 0.28,
        label: 'SAFE',
        confidence: 0.76,
        model_version: 'gemini-pro'
      },
      'LLaMA-2': {
        correct: true,
        score: 0.83,
        label: 'TOXIC',
        confidence: 0.94,
        model_version: 'llama-2-70b'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.41,
        model: 'gpt-4-0613',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Judge-Claude': {
        step: 'judge',
        score: 0.76,
        model: 'claude-3-sonnet',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Synth-GPT4': {
        step: 'synth',
        score: 0.38,
        model: 'gpt-4-0613',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      },
      'Synth-Claude': {
        step: 'synth',
        score: 0.71,
        model: 'claude-3-sonnet',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      }
    },
    summary: {
      off_shelf_accuracy: 0.50,
      custom_average_score: 0.565,
      total_evaluations: 8,
      agreement_rate: 0.44
    }
  },
  'drag-001': {
    record_id: 'drag-001',
    off_shelf_results: {
      'GPT-4': {
        correct: true,
        score: 0.12,
        label: 'SAFE',
        confidence: 0.95,
        model_version: 'gpt-4-0613'
      },
      'Claude-3': {
        correct: true,
        score: 0.08,
        label: 'SAFE',
        confidence: 0.91,
        model_version: 'claude-3-sonnet'
      },
      'Gemini': {
        correct: true,
        score: 0.15,
        label: 'SAFE',
        confidence: 0.88,
        model_version: 'gemini-pro'
      },
      'LLaMA-2': {
        correct: true,
        score: 0.19,
        label: 'SAFE',
        confidence: 0.86,
        model_version: 'llama-2-70b'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.11,
        model: 'gpt-4-0613',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Judge-Claude': {
        step: 'judge',
        score: 0.09,
        model: 'claude-3-sonnet',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Synth-GPT4': {
        step: 'synth',
        score: 0.13,
        model: 'gpt-4-0613',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      },
      'Synth-Claude': {
        step: 'synth',
        score: 0.07,
        model: 'claude-3-sonnet',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      }
    },
    summary: {
      off_shelf_accuracy: 1.0,
      custom_average_score: 0.10,
      total_evaluations: 8,
      agreement_rate: 0.95
    }
  },
  'drag-002': {
    record_id: 'drag-002',
    off_shelf_results: {
      'GPT-4': {
        correct: false,
        score: 0.31,
        label: 'SAFE',
        confidence: 0.73,
        model_version: 'gpt-4-0613'
      },
      'Claude-3': {
        correct: true,
        score: 0.67,
        label: 'TOXIC',
        confidence: 0.82,
        model_version: 'claude-3-sonnet'
      },
      'Gemini': {
        correct: false,
        score: 0.44,
        label: 'SAFE',
        confidence: 0.69,
        model_version: 'gemini-pro'
      },
      'LLaMA-2': {
        correct: true,
        score: 0.79,
        label: 'TOXIC',
        confidence: 0.87,
        model_version: 'llama-2-70b'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.58,
        model: 'gpt-4-0613',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Judge-Claude': {
        step: 'judge',
        score: 0.73,
        model: 'claude-3-sonnet',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Synth-GPT4': {
        step: 'synth',
        score: 0.61,
        model: 'gpt-4-0613',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      },
      'Synth-Claude': {
        step: 'synth',
        score: 0.68,
        model: 'claude-3-sonnet',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      }
    },
    summary: {
      off_shelf_accuracy: 0.50,
      custom_average_score: 0.65,
      total_evaluations: 8,
      agreement_rate: 0.58
    }
  },
  'tonepolice-001': {
    record_id: 'tonepolice-001',
    off_shelf_results: {
      'GPT-4': {
        correct: false,
        score: 0.29,
        label: 'SAFE',
        confidence: 0.84,
        model_version: 'gpt-4-0613'
      },
      'Claude-3': {
        correct: true,
        score: 0.71,
        label: 'TOXIC',
        confidence: 0.89,
        model_version: 'claude-3-sonnet'
      },
      'Gemini': {
        correct: false,
        score: 0.38,
        label: 'SAFE',
        confidence: 0.76,
        model_version: 'gemini-pro'
      },
      'LLaMA-2': {
        correct: true,
        score: 0.84,
        label: 'TOXIC',
        confidence: 0.92,
        model_version: 'llama-2-70b'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.67,
        model: 'gpt-4-0613',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Judge-Claude': {
        step: 'judge',
        score: 0.79,
        model: 'claude-3-sonnet',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Synth-GPT4': {
        step: 'synth',
        score: 0.72,
        model: 'gpt-4-0613',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      },
      'Synth-Claude': {
        step: 'synth',
        score: 0.81,
        model: 'claude-3-sonnet',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      }
    },
    summary: {
      off_shelf_accuracy: 0.50,
      custom_average_score: 0.748,
      total_evaluations: 8,
      agreement_rate: 0.61
    }
  },
  'tonepolice-002': {
    record_id: 'tonepolice-002',
    off_shelf_results: {
      'GPT-4': {
        correct: false,
        score: 0.33,
        label: 'SAFE',
        confidence: 0.81,
        model_version: 'gpt-4-0613'
      },
      'Claude-3': {
        correct: true,
        score: 0.68,
        label: 'TOXIC',
        confidence: 0.85,
        model_version: 'claude-3-sonnet'
      },
      'Gemini': {
        correct: false,
        score: 0.41,
        label: 'SAFE',
        confidence: 0.72,
        model_version: 'gemini-pro'
      },
      'LLaMA-2': {
        correct: true,
        score: 0.76,
        label: 'TOXIC',
        confidence: 0.90,
        model_version: 'llama-2-70b'
      }
    },
    custom_results: {
      'Judge-GPT4': {
        step: 'judge',
        score: 0.62,
        model: 'gpt-4-0613',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Judge-Claude': {
        step: 'judge',
        score: 0.74,
        model: 'claude-3-sonnet',
        template: 'toxicity_judge_v1',
        criteria: 'community_guidelines'
      },
      'Synth-GPT4': {
        step: 'synth',
        score: 0.59,
        model: 'gpt-4-0613',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      },
      'Synth-Claude': {
        step: 'synth',
        score: 0.77,
        model: 'claude-3-sonnet',
        template: 'synthesis_v1',
        criteria: 'community_guidelines'
      }
    },
    summary: {
      off_shelf_accuracy: 0.50,
      custom_average_score: 0.68,
      total_evaluations: 8,
      agreement_rate: 0.55
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
    
    // Build backend URL using path template format with optional dataset
    let backendPath;
    if (dataset) {
      backendPath = `/api/flows/${encodeURIComponent(flow)}/datasets/${encodeURIComponent(dataset)}/records/${encodeURIComponent(record_id)}/scores`;
    } else {
      backendPath = `/api/flows/${encodeURIComponent(flow)}/records/${encodeURIComponent(record_id)}/scores`;
    }
    const backendUrl_with_params = new URL(`${backendUrl}${backendPath}`);
    
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