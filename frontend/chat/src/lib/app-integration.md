# Connecting the Terminal UI to Your Application

This document provides instructions on how to integrate the terminal UI with your existing application.

## Integration Overview

The integration consists of three main parts:
1. **WebSocket Server**: Modify to connect with your application logic
2. **Message Handling**: Format messages from your app to display in the terminal UI
3. **Command Processing**: Process messages from the terminal UI in your application

## 1. WebSocket Server Integration

### Modify the WebSocket handler

The current WebSocket handler in `src/routes/api/websocket/+server.ts` has a simple echo implementation. You need to modify it to connect to your application:

```typescript
// Example modification to connect to your app
import { yourAppMessageHandler } from '$lib/your-app-connector';

export function GET(event: RequestEvent) {
  // ... existing code ...
  
  // Set up message handler for the server-side
  server.addEventListener('message', (event) => {
    try {
      // Parse the message data (form submission from HTMX)
      const formData = new URLSearchParams(event.data);
      const message = formData.get('message');
      
      if (message) {
        // Send the user message as HTML fragment
        const userMessageHtml = formatMessageAsHtml('User', message);
        server.send(userMessageHtml);
        
        // Process message with your application
        yourAppMessageHandler(message, (response) => {
          // Send the response back to the UI
          const appResponseHtml = formatMessageAsHtml('App', response);
          server.send(appResponseHtml);
        });
      }
    } catch (error) {
      // ... error handling ...
    }
  });
  
  // ... rest of the code ...
}

function formatMessageAsHtml(username, message) {
  const timestamp = new Date().toLocaleTimeString();
  return `
    <div class="message">
      <span class="timestamp">[${timestamp}]</span>
      <span class="username">${username}:</span>
      <span class="message-text">${message}</span>
    </div>
  `;
}
```

## 2. Create Your App Connector

Create a new file in `src/lib/your-app-connector.ts` to handle the connection between the UI and your application:

```typescript
// This is a simplified example
// Modify based on your actual application structure

// Store the WebSocket connection for sending messages from your app
let activeWebSocket: WebSocket | null = null;

// Function to set the active WebSocket
export function setActiveWebSocket(ws: WebSocket) {
  activeWebSocket = ws;
  
  // You could trigger an initial connection message here
  sendSystemMessage('Connected to your application');
}

// Handle messages from the UI to your app
export function yourAppMessageHandler(message: string, callback: (response: string) => void) {
  // Here you would process the message with your application logic
  
  // Example: Simulated application processing
  setTimeout(() => {
    // Your app processes the message and returns a response
    const appResponse = processWithYourApp(message);
    
    // Send the response back to the UI via callback
    callback(appResponse);
  }, 500);
}

// Function to send messages from your app to the UI
export function sendMessageFromApp(agentName: string, message: string) {
  if (activeWebSocket && activeWebSocket.readyState === WebSocket.OPEN) {
    const messageHtml = formatMessageAsHtml(agentName, message);
    activeWebSocket.send(messageHtml);
    return true;
  }
  return false;
}

// Helper function for system messages
export function sendSystemMessage(message: string) {
  return sendMessageFromApp('System', message);
}

// Connect to your actual app here
function processWithYourApp(message: string): string {
  // Replace this with your actual app integration
  return `Processing: "${message}"`;
}

// Helper to format messages
function formatMessageAsHtml(username: string, message: string): string {
  const timestamp = new Date().toLocaleTimeString();
  return `
    <div class="message">
      <span class="timestamp">[${timestamp}]</span>
      <span class="username">${username}:</span>
      <span class="message-text">${message}</span>
    </div>
  `;
}
```

## 3. Update the WebSocket Server to Use Your Connector

Modify `src/routes/api/websocket/+server.ts` to use your connector:

```typescript
import { RequestEvent } from '@sveltejs/kit';
import { setActiveWebSocket, yourAppMessageHandler } from '$lib/your-app-connector';

export function GET(event: RequestEvent) {
  // ... existing code ...
  
  // Store the server WebSocket for your app to use
  setActiveWebSocket(server);
  
  // ... rest of your code ...
}
```

## 4. Button Integration

The UI includes two icon buttons for "confirm" and "interrupt". Add event handlers for these:

```typescript
// In your +page.svelte file
<script>
  // ... existing code ...
  
  function handleConfirm() {
    // Send a confirm action to your app
    const messageInput = document.getElementById('message-input');
    const confirmMessage = `__CONFIRM__:${messageInput.value || ''}`;
    
    // Use the existing form's ws-send mechanism or a custom event
    // For example, trigger a custom event that simulates form submission
    const formData = new FormData();
    formData.append('message', confirmMessage);
    
    // Send via the WebSocket connection
    const wsConnection = htmx.find('div[ws-connect]').ws;
    if (wsConnection) {
      wsConnection.send(new URLSearchParams(formData).toString());
    }
  }
  
  function handleInterrupt() {
    // Similar to above, but for interrupt
    const interruptMessage = '__INTERRUPT__';
    
    const formData = new FormData();
    formData.append('message', interruptMessage);
    
    const wsConnection = htmx.find('div[ws-connect]').ws;
    if (wsConnection) {
      wsConnection.send(new URLSearchParams(formData).toString());
    }
  }
</script>

<!-- Then update your button elements with the event handlers -->
<button 
  type="button" 
  class="btn btn-success icon-button confirm-button"
  aria-label="Confirm"
  on:click={handleConfirm}>
  <i class="bi bi-check-circle"></i>
</button>

<button 
  type="button" 
  class="btn btn-danger icon-button interrupt-button"
  aria-label="Interrupt"
  on:click={handleInterrupt}>
  <i class="bi bi-x-circle"></i>
</button>
```

## 5. Processing Special Commands

Update your app connector to handle special commands:

```typescript
export function yourAppMessageHandler(message: string, callback: (response: string) => void) {
  // Check for special commands
  if (message.startsWith('__CONFIRM__:')) {
    const confirmContent = message.replace('__CONFIRM__:', '');
    handleConfirmInYourApp(confirmContent);
    return;
  }
  
  if (message === '__INTERRUPT__') {
    handleInterruptInYourApp();
    return;
  }
  
  // Handle regular messages as before
  // ...
}

function handleConfirmInYourApp(content: string) {
  // Your app-specific logic for confirmation actions
  console.log('Confirmation received:', content);
  // Trigger relevant action in your app
}

function handleInterruptInYourApp() {
  // Your app-specific logic for interrupt actions
  console.log('Interrupt received');
  // Stop current processing or cancel current action in your app
}
```

## Complete Integration Example

For a complete integration example, you would:

1. Create your app connector file
2. Update the WebSocket server to use your connector
3. Add event handlers for the buttons
4. Process messages from your app to display in the UI
5. Process messages from the UI in your app

You can adapt this integration to work with any application by modifying the app connector to interface with your specific application logic.
