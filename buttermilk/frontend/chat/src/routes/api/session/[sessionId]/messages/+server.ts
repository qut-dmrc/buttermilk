import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, url }) => {
  const { sessionId } = params;
  
  // Get backend URL from environment
  const backendUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
  
  try {
    // Forward request to backend
    const backendResponse = await fetch(`${backendUrl}/api/session/${sessionId}/messages`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!backendResponse.ok) {
      console.error(`Backend error: ${backendResponse.status} ${backendResponse.statusText}`);
      return json(
        { error: 'Failed to fetch session messages' }, 
        { status: backendResponse.status }
      );
    }
    
    const data = await backendResponse.json();
    return json(data);
    
  } catch (error) {
    console.error('Error proxying session messages request:', error);
    return json(
      { error: 'Internal server error' }, 
      { status: 500 }
    );
  }
};