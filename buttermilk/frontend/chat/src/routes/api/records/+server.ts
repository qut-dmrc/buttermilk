import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

// Sample record data for fallback when backend is unavailable
const FALLBACK_RECORDS = {
  'osb': [
    { 
      id: 'osb-001', 
      name: 'Oversight Board Case #001', 
      description: 'Platform moderation policy discussion',
      summary_scores: {
        off_shelf_accuracy: 0.75,
        custom_average: 0.635,
        total_evaluations: 8,
        has_detailed_responses: true
      }
    },
    { 
      id: 'osb-002', 
      name: 'Oversight Board Case #002', 
      description: 'Satirical content with political commentary',
      summary_scores: {
        off_shelf_accuracy: 0.50,
        custom_average: 0.565,
        total_evaluations: 8,
        has_detailed_responses: true
      }
    }
  ],
  'drag': [
    { 
      id: 'drag-001', 
      name: 'Drag Performance Analysis #001', 
      description: 'Positive drag culture celebration',
      summary_scores: {
        off_shelf_accuracy: 1.0,
        custom_average: 0.10,
        total_evaluations: 8,
        has_detailed_responses: true
      }
    },
    { 
      id: 'drag-002', 
      name: 'Drag Performance Analysis #002', 
      description: 'Suspected coded messaging in drag context',
      summary_scores: {
        off_shelf_accuracy: 0.50,
        custom_average: 0.65,
        total_evaluations: 8,
        has_detailed_responses: true
      }
    }
  ],
  'tonepolice': [
    { 
      id: 'tonepolice-001', 
      name: 'Tone Policing Example #001', 
      description: 'Civility-focused dismissal of social justice concerns',
      summary_scores: {
        off_shelf_accuracy: 0.50,
        custom_average: 0.748,
        total_evaluations: 8,
        has_detailed_responses: true
      }
    },
    { 
      id: 'tonepolice-002', 
      name: 'Tone Policing Example #002', 
      description: 'Emotional tone dismissal of systemic criticism',
      summary_scores: {
        off_shelf_accuracy: 0.50,
        custom_average: 0.68,
        total_evaluations: 8,
        has_detailed_responses: true
      }
    }
  ],
  // Legacy fallback data for compatibility
  'user-authentication': [
    { id: 'auth-1', name: 'Login Success Rate', description: 'Percentage of successful logins' },
    { id: 'auth-2', name: 'Failed Attempts', description: 'Number of failed login attempts' },
    { id: 'auth-3', name: 'Account Lockouts', description: 'Number of accounts locked due to failed attempts' },
    { id: 'auth-4', name: 'Password Resets', description: 'Number of password reset requests' }
  ],
  'content-moderation': [
    { id: 'mod-1', name: 'Flagged Content', description: 'Content flagged as potentially violating guidelines' },
    { id: 'mod-2', name: 'Removed Posts', description: 'Posts that were removed for violating guidelines' },
    { id: 'mod-3', name: 'User Reports', description: 'Reports submitted by users about content' },
    { id: 'mod-4', name: 'False Positives', description: 'Content incorrectly flagged as violating' }
  ],
  'data-extraction': [
    { id: 'extract-1', name: 'Documents Processed', description: 'Number of documents processed' },
    { id: 'extract-2', name: 'Extraction Accuracy', description: 'Accuracy rate of data extraction' },
    { id: 'extract-3', name: 'Processing Time', description: 'Average time to process a document' }
  ],
  'sentiment-analysis': [
    { id: 'sent-1', name: 'Positive Sentiment', description: 'Content with positive sentiment' },
    { id: 'sent-2', name: 'Negative Sentiment', description: 'Content with negative sentiment' },
    { id: 'sent-3', name: 'Neutral Sentiment', description: 'Content with neutral sentiment' },
    { id: 'sent-4', name: 'Confidence Score', description: 'Average confidence score of sentiment analysis' }
  ],
  'entity-recognition': [
    { id: 'ent-1', name: 'People', description: 'Recognized people entities' },
    { id: 'ent-2', name: 'Organizations', description: 'Recognized organization entities' },
    { id: 'ent-3', name: 'Locations', description: 'Recognized location entities' },
    { id: 'ent-4', name: 'Dates', description: 'Recognized date entities' },
    { id: 'ent-5', name: 'Custom Entities', description: 'User-defined entity types' }
  ]
};

export const GET: RequestHandler = async ({ fetch, request, url }) => {
  // Get query parameters
  const flow = url.searchParams.get('flow');
  const dataset = url.searchParams.get('dataset');
  const includeScores = url.searchParams.get('include_scores') === 'true';
  
  // If no flow is specified, return an empty array
  if (!flow) {
    return json([]);
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
    
    // Build backend URL with query parameters
    const backendQueryParams = new URLSearchParams();
    backendQueryParams.append('flow', flow);
    if (dataset) {
      backendQueryParams.append('dataset', dataset);
    }
    if (includeScores) {
      backendQueryParams.append('include_scores', 'true');
    }
    
    // Forward the request to the backend
    const response = await fetch(`${backendUrl}/api/records?${backendQueryParams.toString()}`, {
      method: 'GET',
      headers
    });
    
    if (response.ok) {
      const data = await response.json();
      return json(data);
    }
    
    // If backend is unavailable or returns error, fall back to mock data
    console.warn(`Backend unavailable for records flow=${flow}, using fallback data`);
    
  } catch (error) {
    console.warn('Error fetching records from backend:', error);
  }
  
  // Return fallback data if available
  if (FALLBACK_RECORDS[flow as keyof typeof FALLBACK_RECORDS]) {
    let records = FALLBACK_RECORDS[flow as keyof typeof FALLBACK_RECORDS];
    
    // If scores aren't requested, remove them from the response
    if (!includeScores) {
      records = records.map((record: any) => {
        const { summary_scores, ...recordWithoutScores } = record;
        return recordWithoutScores;
      });
    }
    
    return json(records);
  }
  
  // No fallback data available
  return json([]);
};
