/**
 * Example App Connector
 * 
 * This is a simplified example of how to connect your application to the terminal UI.
 * Replace the placeholder implementations with your actual application logic.
 */

// Store the WebSocket connection for sending messages from your app
let activeWebSocket: any = null;

// Function to set the active WebSocket
export function setActiveWebSocket(ws: any) {
  activeWebSocket = ws;
  
  // Send an initial connection message
  sendSystemMessage('Connected to example application');
  
  // Demo: Simulate periodic messages from your app
  startDemoMessages();
}

// Handle messages from the UI to your app
export function processMessage(message: string, callback: (response: string) => void) {
  console.log('Received message from UI:', message);
  
  // Check for special commands
  if (message.startsWith('__CONFIRM__:')) {
    const confirmContent = message.replace('__CONFIRM__:', '');
    handleConfirm(confirmContent);
    return;
  }
  
  if (message === '__INTERRUPT__') {
    handleInterrupt();
    return;
  }
  
  // Process regular messages
  // This is where you'd integrate with your actual application logic
  setTimeout(() => {
    const response = `App received: "${message}"`;
    callback(response);
    
    // Simulate a follow-up message from a different agent
    setTimeout(() => {
      sendMessageFromApp('Agent', 'Processing your request...');
    }, 1000);
  }, 500);
}

// Function to send messages from your app to the UI
export function sendMessageFromApp(agentName: string, message: string) {
  if (activeWebSocket && activeWebSocket.readyState === 1) { // WebSocket.OPEN
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

// Handle confirm button clicks
function handleConfirm(content: string) {
  console.log('Confirmation received:', content);
  sendSystemMessage(`Confirmed: ${content}`);
  
  // Your app-specific confirmation logic here
}

// Handle interrupt button clicks
function handleInterrupt() {
  console.log('Interrupt received');
  sendSystemMessage('Operation interrupted');
  
  // Your app-specific interruption logic here
}

// Helper to format messages as HTML for the UI
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
