import { WebSocketServer, WebSocket } from 'ws';
import { Message } from '../types.js';

export interface MockServerOptions {
  port: number;
  sessionId?: string;
  scenarioFile?: string;
  autoResponses?: Map<string, Message[]>;
  delayMs?: number;
}

export class MockWebSocketServer {
  private wss: WebSocketServer | null = null;
  private clients: Set<WebSocket> = new Set();
  private messageLog: Array<{ type: 'sent' | 'received', message: Message, timestamp: Date }> = [];
  private autoResponses: Map<string, Message[]>;
  private delayMs: number;
  private sessionId: string;

  constructor(private options: MockServerOptions) {
    this.autoResponses = options.autoResponses || new Map();
    this.delayMs = options.delayMs || 100;
    this.sessionId = options.sessionId || 'test-session-123';
  }

  async start(): Promise<void> {
    return new Promise((resolve) => {
      this.wss = new WebSocketServer({ port: this.options.port });
      
      this.wss.on('connection', (ws, req) => {
        console.log(`[MockServer] Client connected to ${req.url}`);
        this.clients.add(ws);
        
        // Send initial connection message
        this.sendMessage(ws, {
          type: 'system_message',
          payload: { message: 'Connected to mock server' }
        });

        ws.on('message', async (data) => {
          try {
            const message = JSON.parse(data.toString()) as Message;
            this.logMessage('received', message);
            console.log('[MockServer] Received:', message);
            
            // Process message and generate responses
            await this.handleMessage(ws, message);
          } catch (error) {
            console.error('[MockServer] Error parsing message:', error);
          }
        });

        ws.on('close', () => {
          console.log('[MockServer] Client disconnected');
          this.clients.delete(ws);
        });

        ws.on('error', (error) => {
          console.error('[MockServer] WebSocket error:', error);
        });
      });

      this.wss.on('listening', () => {
        console.log(`[MockServer] WebSocket server listening on port ${this.options.port}`);
        resolve();
      });
    });
  }

  private async handleMessage(ws: WebSocket, message: Message): Promise<void> {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, this.delayMs));

    switch (message.type) {
      case 'run_flow':
        await this.handleRunFlow(ws, message);
        break;
      
      case 'user_message':
        await this.handleUserMessage(ws, message);
        break;
      
      case 'manager_response':
        await this.handleManagerResponse(ws, message);
        break;
      
      default:
        // Check for auto-responses
        const responses = this.autoResponses.get(message.type);
        if (responses) {
          for (const response of responses) {
            await this.sendMessage(ws, response);
          }
        }
    }
  }

  private async handleRunFlow(ws: WebSocket, message: Message): Promise<void> {
    const flowName = message.payload?.flow;
    const prompt = message.payload?.prompt;
    
    // Send flow start acknowledgment
    await this.sendMessage(ws, {
      type: 'flow_progress_update',
      payload: {
        step_name: 'initialization',
        status: 'STARTED',
        message: `Starting flow: ${flowName}`
      }
    });

    // Simulate different flow scenarios
    switch (flowName) {
      case 'test':
        await this.runTestFlow(ws);
        break;
      
      case 'osb':
        await this.runOSBFlow(ws, prompt);
        break;
      
      default:
        await this.sendMessage(ws, {
          type: 'system_error',
          payload: { message: `Unknown flow: ${flowName}` }
        });
    }
  }

  private async runTestFlow(ws: WebSocket): Promise<void> {
    // Simulate agent joining
    await this.sendMessage(ws, {
      type: 'agent_announcement',
      payload: {
        agent_id: 'test_agent',
        action: 'joined',
        status_message: 'Test agent ready'
      }
    });

    // Simulate some work
    await this.sendMessage(ws, {
      type: 'agent_output',
      payload: {
        agent_id: 'test_agent',
        content: 'Running test operations...'
      }
    });

    // Complete
    await this.sendMessage(ws, {
      type: 'flow_progress_update',
      payload: {
        step_name: 'completion',
        status: 'COMPLETED',
        message: 'Test flow completed successfully'
      }
    });
  }

  private async runOSBFlow(ws: WebSocket, prompt?: string): Promise<void> {
    // Announce agents
    const agents = ['researcher', 'policy_analyst', 'fact_checker', 'explorer'];
    for (const agent of agents) {
      await this.sendMessage(ws, {
        type: 'agent_announcement',
        payload: {
          agent_id: agent,
          action: 'joined'
        }
      });
    }

    // Send UI message for confirmation
    await this.sendMessage(ws, {
      type: 'ui_message',
      payload: {
        message: `Ready to analyze: "${prompt}". Continue?`,
        requires_response: true
      }
    });

    // Wait for manager response...
  }

  private async handleManagerResponse(ws: WebSocket, message: Message): Promise<void> {
    const response = message.payload?.text || message.payload?.response;
    
    if (response?.toLowerCase() === 'yes') {
      // Continue OSB flow
      await this.sendMessage(ws, {
        type: 'osb_status',
        payload: {
          message: 'Starting multi-agent analysis...'
        }
      });

      // Simulate agent responses
      await this.sendMessage(ws, {
        type: 'osb_partial',
        payload: {
          agent: 'researcher',
          content: 'Mock research findings about the topic...'
        }
      });

      await this.sendMessage(ws, {
        type: 'osb_complete',
        payload: {
          content: 'Mock synthesis of all agent findings...',
          recommendations: ['Recommendation 1', 'Recommendation 2']
        }
      });
    }
  }

  private async handleUserMessage(ws: WebSocket, message: Message): Promise<void> {
    // Echo back with a response
    await this.sendMessage(ws, {
      type: 'system_message',
      payload: {
        message: `Mock server received: ${message.payload?.text}`
      }
    });
  }

  private async sendMessage(ws: WebSocket, message: Message): Promise<void> {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      this.logMessage('sent', message);
      console.log('[MockServer] Sent:', message);
      
      // Add delay between messages
      await new Promise(resolve => setTimeout(resolve, this.delayMs));
    }
  }

  private logMessage(type: 'sent' | 'received', message: Message): void {
    this.messageLog.push({
      type,
      message,
      timestamp: new Date()
    });
  }

  broadcast(message: Message): void {
    this.clients.forEach(client => {
      this.sendMessage(client, message);
    });
  }

  getMessageLog(): typeof this.messageLog {
    return [...this.messageLog];
  }

  async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.wss) {
        this.clients.forEach(client => client.close());
        this.wss.close(() => {
          console.log('[MockServer] Server stopped');
          resolve();
        });
      } else {
        resolve();
      }
    });
  }
}

// HTTP server for session endpoint
import http from 'http';

export class MockHTTPServer {
  private server: http.Server | null = null;
  
  constructor(private port: number, private sessionId: string) {}

  start(): Promise<void> {
    return new Promise((resolve) => {
      this.server = http.createServer((req, res) => {
        if (req.url === '/api/session' && req.method === 'GET') {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ session_id: this.sessionId }));
        } else {
          res.writeHead(404);
          res.end('Not found');
        }
      });

      this.server.listen(this.port, () => {
        console.log(`[MockHTTP] Server listening on port ${this.port}`);
        resolve();
      });
    });
  }

  stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => {
          console.log('[MockHTTP] Server stopped');
          resolve();
        });
      } else {
        resolve();
      }
    });
  }
}