import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

// Sample record data for fallback when backend is unavailable
const FALLBACK_RECORDS = {
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
    
    // Forward the request to the backend with the flow parameter
    const response = await fetch(`${backendUrl}/api/flowinfo?flow=${encodeURIComponent(flow)}`, {
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
