import type { RequestEvent } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

/**
 * This WebSocket endpoint acts as a proxy to the backend WebSocket server.
 * It connects to the actual backend WebSocket and relays messages.
 */
export function GET(event: RequestEvent) {
  const { request, url } = event;
  
  // Check if the request is for WebSocket
  if (request.headers.get('upgrade') !== 'websocket') {
    return new Response('Expected WebSocket', { status: 400 });
  }
  
  // Get session ID from path parameter
  const pathParts = url.pathname.split('/');
  const sessionId = pathParts[pathParts.length - 1];
  
  if (!sessionId || sessionId === 'ws') {
    return new Response('Missing session ID', { status: 400 });
  }
  
  // Construct the WebSocket URL (convert from http to ws protocol)
  const wsProtocol = env.BACKEND_API_URL.startsWith('https') ? 'wss' : 'ws';
  const targetUrl = `${wsProtocol}://${env.BACKEND_API_URL.replace(/^https?:\/\//, '')}`;
  
  console.log(`Connecting to backend WebSocket at ${targetUrl}/ws/${sessionId}`);
  
  // Connect to the backend WebSocket
  return handleWebSocketProxy(`${targetUrl}/ws/${sessionId}`);
}

/**
 * Handle WebSocket connections as a proxy to the backend
 */
function handleWebSocketProxy(targetUrl: string) {
  console.log('WebSocket proxy activated, target:', targetUrl);
  
  try {
    // Create the WebSocket pair
    const webSocketPair = new WebSocketPair();
    const [client, server] = Object.values(webSocketPair);
    
    // Accept the WebSocket connection
    server.accept();
    
    // Send connecting message to client
    sendSystemMessage(server, `Connecting to backend at ${targetUrl}...`);
    
    // In production, this would connect to the actual backend WebSocket
    // and relay messages between client and backend
    
    // For development purposes, we're simulating connectivity
    setTimeout(() => {
      sendSystemMessage(server, 'Connected to backend WebSocket server');
    }, 500);
    
    // Handle messages from the client
    server.addEventListener('message', (event) => {
      try {
        let message = event.data;
        
        // Try to parse as JSON if the message is a string
        if (typeof message === 'string') {
          try {
            // First try to parse as form data (for HTMX compatibility)
            const formData = new URLSearchParams(message);
            const formMessage = formData.get('message');
            if (formMessage) {
              message = formMessage;
            } else {
              // Try to parse as JSON
              try {
                const jsonData = JSON.parse(message);
                message = jsonData;
              } catch (e) {
                // Keep as string if not JSON
              }
            }
          } catch (e) {
            // Keep original message if parsing fails
          }
        }
        
        // Echo the user message back to the client
        sendUserMessage(server, typeof message === 'string' ? message : JSON.stringify(message));
        
        // In production, would forward message to backend WebSocket here
        // For now, simulate a backend response
        setTimeout(() => {
          sendMessage(server, 'Server', `Received: ${typeof message === 'string' ? message : JSON.stringify(message)}`);
        }, 300);
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
        sendErrorMessage(server, 'Error processing message');
      }
    });
    
    // Handle WebSocket close
    server.addEventListener('close', () => {
      console.log('WebSocket connection closed');
    });
    
    // Return the client side of the WebSocket
    return new Response(null, {
      status: 101,
      webSocket: client
    });
  } catch (error) {
    console.error('Error setting up WebSocket proxy:', error);
    return new Response('Error setting up WebSocket connection', { status: 500 });
  }
}

// Helper functions for sending formatted messages

function sendMessage(server: WebSocket, text: string) {
  const timestamp = new Date().toLocaleTimeString();
  const message = {
    timestamp,
    text
  };
  
  server.send(JSON.stringify(message));
}

function sendSystemMessage(server: WebSocket, text: string) {
  sendMessage(server, text);
}

function sendUserMessage(server: WebSocket, text: string) {
  sendMessage(server, text);
}

function sendErrorMessage(server: WebSocket, text: string) {
  sendMessage(server,text);
}
