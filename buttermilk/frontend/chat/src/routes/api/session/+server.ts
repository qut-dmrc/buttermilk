import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

// Define interface for session data
interface SessionData {
  sessionId: string;
  new?: boolean;
  [key: string]: any; // Allow for additional properties from backend
}

export const GET: RequestHandler = async ({ fetch, url }) => {
  try {
    // Check if an existing session ID was provided in the request
    const existingSessionId = url.searchParams.get('existingSessionId');
    
    if (existingSessionId) {
      console.log(`Validating existing session ID: ${existingSessionId}`);
      
      // Attempt to validate the existing session ID with the backend
      // Note: Backend would need a validation endpoint. For now, we'll assume if the session
      // exists, it's valid. In a production environment, this should properly validate.
      try {
        // You'd normally have a validation endpoint like:
        // const validateResponse = await fetch(`${env.BACKEND_API_URL}/api/session/validate?sessionId=${existingSessionId}`);
        
        // For now, let's use the session directly and assume it's valid if we don't get an error
        const validateResponse = await fetch(`${env.BACKEND_API_URL}/api/session?sessionId=${existingSessionId}`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json'
          }
        });
        
        if (validateResponse.ok) {
          console.log('Existing session is valid, reusing it');
          // If validation succeeded, return the existing session ID
          return json({ 
            sessionId: existingSessionId, 
            new: false 
          });
        } else {
          console.log('Existing session validation failed, will create new session');
          // If validation failed, fall through to create a new session
        }
      } catch (validationError) {
        console.error('Error validating session:', validationError);
        // If validation throws an error, fall through to create a new session
      }
    }
    
    // Get a new session ID from the backend
    console.log('Requesting new session from backend: ${env.BACKEND_API_URL}/api/session');
    const response = await fetch(`${env.BACKEND_API_URL}/api/session`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get session from backend: ${response.statusText}`);
    }
    
    const sessionData = await response.json() as SessionData;
    
    // Mark this as a new session
    sessionData.new = true;
    
    // Return the session data
    return json(sessionData);
  } catch (error) {
    console.error('Error fetching session from backend:', error);
    return new Response(JSON.stringify({ 
      error: 'Failed to get session from backend',
      details: error instanceof Error ? error.message : 'Unknown error'
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
};

// Note: In future, we might need to add:
// 1. A proper session validation endpoint on the backend
// 2. A session cleanup endpoint for logout functionality
