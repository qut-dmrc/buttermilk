import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

// Mock responses data based on the API spec for fallback
const MOCK_RESPONSES: Record<string, any> = {
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
      },
      {
        agent: 'Judge-Claude',
        type: 'judge',
        model: 'claude-3-sonnet',
        content: 'While the content discusses sensitive topics, it appears to be educational in nature.',
        score: 0.45,
        reasoning: 'The context suggests academic analysis rather than promoting harmful behavior. The language, while discussing difficult topics, maintains an analytical tone and appears to be questioning platform policies rather than advocating for harmful content.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-15T10:31:00Z',
        confidence: 0.78,
        prediction: false
      },
      {
        agent: 'Synth-GPT4',
        type: 'synth',
        model: 'gpt-4-0613',
        content: 'Analysis reveals concerning patterns in language choice and framing.',
        score: 0.82,
        reasoning: 'Synthesis of multiple evaluation perspectives indicates subtle but present toxicity markers. The content uses techniques commonly seen in borderline violations.',
        criteria_used: 'community_guidelines',
        template: 'synthesis_v1',
        timestamp: '2024-01-15T10:32:00Z',
        confidence: 0.85,
        prediction: true
      },
      {
        agent: 'Synth-Claude',
        type: 'synth',
        model: 'claude-3-sonnet',
        content: 'Content appears to be within acceptable bounds with legitimate concerns raised.',
        score: 0.39,
        reasoning: 'Comprehensive analysis suggests the content, while addressing sensitive topics, does so in a way that promotes dialogue rather than harm.',
        criteria_used: 'community_guidelines',
        template: 'synthesis_v1',
        timestamp: '2024-01-15T10:33:00Z',
        confidence: 0.71,
        prediction: false
      }
    ]
  },
  'osb-002': {
    record_id: 'osb-002',
    responses: [
      {
        agent: 'Judge-GPT4',
        type: 'judge',
        model: 'gpt-4-0613',
        content: 'The satirical nature of this content makes it challenging to evaluate for policy violations.',
        score: 0.41,
        reasoning: 'While the content uses exaggerated stereotypes, the satirical framing and political commentary context suggest it may not constitute a direct violation.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-16T14:30:00Z',
        confidence: 0.67,
        prediction: false
      },
      {
        agent: 'Judge-Claude',
        type: 'judge',
        model: 'claude-3-sonnet',
        content: 'Despite satirical intent, this content perpetuates harmful stereotypes.',
        score: 0.76,
        reasoning: 'The use of satirical framing does not mitigate the potential harm from reinforcing negative stereotypes, particularly when consumed by audiences who may not recognize the satirical intent.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-16T14:31:00Z',
        confidence: 0.84,
        prediction: true
      }
    ]
  },
  'drag-001': {
    record_id: 'drag-001',
    responses: [
      {
        agent: 'Judge-GPT4',
        type: 'judge',
        model: 'gpt-4-0613',
        content: 'This content celebrates artistic expression and promotes positive representation.',
        score: 0.11,
        reasoning: 'The content focuses on artistic merit and positive community impact, with no indicators of policy violations or harmful rhetoric.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-17T16:50:00Z',
        confidence: 0.96,
        prediction: false
      },
      {
        agent: 'Judge-Claude',
        type: 'judge',
        model: 'claude-3-sonnet',
        content: 'Content appropriately highlights cultural contributions and artistic value.',
        score: 0.09,
        reasoning: 'This post demonstrates positive engagement with LGBTQ+ culture and artistic expression without any concerning language or problematic framing.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-17T16:51:00Z',
        confidence: 0.94,
        prediction: false
      }
    ]
  },
  'drag-002': {
    record_id: 'drag-002',
    responses: [
      {
        agent: 'Judge-GPT4',
        type: 'judge',
        model: 'gpt-4-0613',
        content: 'While ostensibly about drag culture, this content contains concerning undertones.',
        score: 0.58,
        reasoning: 'The content appears to use discussion of drag culture as a vehicle for subtly promoting problematic ideologies through coded language and selective framing.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-18T09:20:00Z',
        confidence: 0.73,
        prediction: true
      },
      {
        agent: 'Judge-Claude',
        type: 'judge',
        model: 'claude-3-sonnet',
        content: 'Content exhibits markers of covert harmful messaging despite surface-level topic.',
        score: 0.73,
        reasoning: 'Analysis reveals patterns consistent with coded messaging techniques used to promote harmful ideologies while maintaining plausible deniability.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-18T09:21:00Z',
        confidence: 0.81,
        prediction: true
      }
    ]
  },
  'tonepolice-001': {
    record_id: 'tonepolice-001',
    responses: [
      {
        agent: 'Judge-GPT4',
        type: 'judge',
        model: 'gpt-4-0613',
        content: 'This response demonstrates classic tone policing patterns that dismiss legitimate concerns.',
        score: 0.67,
        reasoning: 'The content focuses on delivery rather than substance, effectively silencing valid criticism by imposing civility standards that ignore the urgency of the issues raised.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-19T11:35:00Z',
        confidence: 0.88,
        prediction: true
      },
      {
        agent: 'Judge-Claude',
        type: 'judge',
        model: 'claude-3-sonnet',
        content: 'Content exhibits tone policing behaviors that undermine meaningful dialogue.',
        score: 0.79,
        reasoning: 'This represents a clear example of tone policing, where the focus on manner of expression serves to derail substantive discussion of important social justice issues.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-19T11:36:00Z',
        confidence: 0.91,
        prediction: true
      }
    ]
  },
  'tonepolice-002': {
    record_id: 'tonepolice-002',
    responses: [
      {
        agent: 'Judge-GPT4',
        type: 'judge',
        model: 'gpt-4-0613',
        content: 'Content dismisses valid criticism by focusing on perceived emotional tone.',
        score: 0.62,
        reasoning: 'This response exemplifies tone policing by deflecting from systemic issues to focus on the emotional expression of those affected by these issues.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-20T13:50:00Z',
        confidence: 0.83,
        prediction: true
      },
      {
        agent: 'Judge-Claude',
        type: 'judge',
        model: 'claude-3-sonnet',
        content: 'Clear example of tone policing that prioritizes comfort over justice.',
        score: 0.74,
        reasoning: 'The content demonstrates how tone policing functions to maintain status quo by requiring marginalized voices to moderate their expression when addressing injustice.',
        criteria_used: 'community_guidelines',
        template: 'toxicity_judge_v1',
        timestamp: '2024-01-20T13:51:00Z',
        confidence: 0.87,
        prediction: true
      }
    ]
  }
};

export const GET: RequestHandler = async ({ params, fetch, request, url }) => {
  const { record_id, flow } = params;
  
  if (!record_id) {
    throw error(400, 'Record ID is required');
  }
  
  if (!flow) {
    throw error(400, 'Flow parameter is required');
  }
  
  // Get query parameters
  const dataset = url.searchParams.get('dataset');
  const include_reasoning = url.searchParams.get('include_reasoning') !== 'false'; // Default true
  
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
      backendPath = `/api/flows/${encodeURIComponent(flow)}/datasets/${encodeURIComponent(dataset)}/records/${encodeURIComponent(record_id)}/responses`;
    } else {
      backendPath = `/api/flows/${encodeURIComponent(flow)}/records/${encodeURIComponent(record_id)}/responses`;
    }
    
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