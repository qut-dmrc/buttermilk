class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 0; // CONNECTING
    this.events = {};
    
    // Load the captured messages
    this.messageQueue = [];
    this.responseDelay = 100; // ms between messages
    
    // Auto-connect
    setTimeout(() => {
      this.readyState = 1; // OPEN
      if (this.onopen) this.onopen({ type: 'open' });
      if (this.events['open']) this.events['open'].forEach(cb => cb({ type: 'open' }));
      
      // Start sending messages from queue after connection
      this._processQueue();
    }, 50);
  }

  // Add event listener (WebSocket API compatible)
  addEventListener(event, callback) {
    if (!this.events[event]) this.events[event] = [];
    this.events[event].push(callback);
  }

  // Remove event listener (WebSocket API compatible)
  removeEventListener(event, callback) {
    if (this.events[event]) {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
  }

  // Send message (WebSocket API compatible)
  send(data) {
    console.log('MockWebSocket sent:', data);
    
    // Look up appropriate response based on sent message
    const message = JSON.parse(data);
    
    // Find matching response in our captured data
    if (message.type === 'run_flow') {
      this._queueFlowResponses(message);
    } else if (message.type === 'user_input') {
      this._queueUserInputResponses(message);
    } else if (message.type === 'confirm') {
      this._queueConfirmResponses();
    }
  }

  // Close connection (WebSocket API compatible)
  close() {
    this.readyState = 3; // CLOSED
    if (this.onclose) this.onclose({ type: 'close' });
    if (this.events['close']) this.events['close'].forEach(cb => cb({ type: 'close' }));
  }

  // Load message data from HAR file or parsed JSON
  loadMessages(messages) {
    this.capturedMessages = messages;
  }

  // Process the message queue with delays to simulate real timing
  _processQueue() {
    if (this.messageQueue.length === 0) return;
    
    const message = this.messageQueue.shift();
    
    setTimeout(() => {
      if (this.readyState === 1) { // Only if still open
        if (this.onmessage) this.onmessage({ data: JSON.stringify(message) });
        if (this.events['message']) {
          this.events['message'].forEach(cb => cb({ data: JSON.stringify(message) }));
        }
        this._processQueue(); // Process next message
      }
    }, this.responseDelay);
  }

  // Queue responses for flow run requests
  _queueFlowResponses(message) {
    // Find messages for this flow in our capture
    const flowName = message.flow;
    const recordId = message.record_id;
    
    // Add flow started confirmation
    this.messageQueue.push({
      type: "flow_started",
      flow: flowName,
      record_id: recordId
    });
    
    // Add sequence of messages that would normally come from the server
    // These would be loaded from your captured HAR file
    for (const msg of this._getFlowMessages(flowName, recordId)) {
      this.messageQueue.push(msg);
    }
  }

  // Get messages for a specific flow
  _getFlowMessages(flowName, recordId) {
    // This is where you'd filter messages from your HAR file based on flow and record
    // For now, returning sample messages
    return this.capturedMessages.filter(msg => 
      msg.flowContext && msg.flowContext.flow === flowName && 
      msg.flowContext.recordId === recordId
    );
  }

  // Queue responses for user input
  _queueUserInputResponses(message) {
    this.messageQueue.push({
      type: "message_sent",
      message: message.message
    });
    
    // Add any responses to this input from your captured data
    // ...
  }

  // Queue responses for confirm action
  _queueConfirmResponses() {
    this.messageQueue.push({
      type: "confirmed"
    });
    
    // Add any follow-up messages that occur after confirmation
    // ...
  }
}

// Helper to parse HAR file WebSocket messages
function parseHarFileWebSockets(harData) {
  const messages = [];
  
  try {
    const entries = harData.log.entries;
    
    // Find WebSocket handshake and frames
    for (const entry of entries) {
      // WebSocket frames are in _webSocketMessages
      if (entry._webSocketMessages && Array.isArray(entry._webSocketMessages)) {
        for (const wsMessage of entry._webSocketMessages) {
          try {
            // Try to parse the message content
            const data = JSON.parse(wsMessage.data);
            
            // Add metadata about direction
            messages.push({
              ...data,
              _meta: {
                time: wsMessage.time,
                direction: wsMessage.type // 'send' or 'receive'
              }
            });
          } catch (e) {
            console.warn('Could not parse WebSocket message:', wsMessage.data);
          }
        }
      }
    }
  } catch (e) {
    console.error('Error parsing HAR file:', e);
  }
  
  return messages;
}

// Export both the mock WebSocket and the parser
window.MockWebSocket = MockWebSocket;
window.parseHarFileWebSockets = parseHarFileWebSockets;