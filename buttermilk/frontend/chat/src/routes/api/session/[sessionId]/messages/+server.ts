import { json } from '@sveltejs/kit';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, url }) => {
  const { sessionId } = params;
  
  // Get backend URL from environment
  const backendUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
  
  try {
    // Try to forward request to backend first
    const backendResponse = await fetch(`${backendUrl}/api/session/${sessionId}/messages`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(3000) // 3 second timeout
    });
    
    if (backendResponse.ok) {
      const data = await backendResponse.json();
      return json(data);
    }
    
    console.log(`Backend unavailable (${backendResponse.status}), falling back to file system`);
    
  } catch (error) {
    console.log('Backend unavailable, falling back to file system:', error);
  }
  
  // Fallback: Read directly from file system (demo mode)
  try {
    const sessionsDir = join(process.cwd(), '../../../data/sessions');
    const sessionFile = join(sessionsDir, `${sessionId}.json`);
    
    const fileContent = await readFile(sessionFile, 'utf-8');
    const sessionData = JSON.parse(fileContent);
    
    // Transform to match backend API format
    const response = {
      messages: sessionData.messages || [],
      session_metadata: {
        flow_status: sessionData.flow_status || 'completed',
        is_stale: false,
        is_resumable: false, // Always false for demo mode
        message_count: sessionData.messages ? sessionData.messages.length : 0,
        parameters: sessionData.parameters || {}
      }
    };
    
    return json(response);
    
  } catch (fileError) {
    console.error('Error reading session file:', fileError);
    return json(
      { error: 'Session not found' }, 
      { status: 404 }
    );
  }
};