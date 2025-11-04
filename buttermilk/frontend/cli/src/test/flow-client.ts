#!/usr/bin/env node
import WebSocket from 'ws';
import http from 'http';
import readline from 'readline';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

class FlowClient {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;

  constructor(
    private host: string,
    private port: number
  ) {}

  async getSessionId(): Promise<string> {
    return new Promise((resolve, reject) => {
      const req = http.request({
        hostname: this.host,
        port: this.port,
        path: '/api/session',
        method: 'GET'
      }, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            const json = JSON.parse(data);
            resolve(json.session_id);
          } catch (e) {
            reject(e);
          }
        });
      });
      req.on('error', reject);
      req.end();
    });
  }

  async connect(): Promise<void> {
    console.log(`🔌 Connecting to ${this.host}:${this.port}...`);

    this.sessionId = await this.getSessionId();
    console.log(`📋 Session ID: ${this.sessionId}`);

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`ws://${this.host}:${this.port}/ws/${this.sessionId}`);

      this.ws.on('open', () => {
        console.log('✅ Connected!\n');
        resolve();
      });

      this.ws.on('message', (data) => {
        const msg = JSON.parse(data.toString());
        this.handleMessage(msg);
      });

      this.ws.on('error', (error) => {
        console.error('❌ WebSocket error:', error);
        reject(error);
      });

      this.ws.on('close', () => {
        console.log('\n🔌 Disconnected');
        process.exit(0);
      });
    });
  }

  private handleMessage(msg: any): void {
    const timestamp = new Date().toLocaleTimeString();

    console.log(`\n[${timestamp}] 📨 ${msg.type}`);

    switch (msg.type) {
      case 'flow_progress_update':
        const status = msg.payload?.status || 'UNKNOWN';
        const icon = status === 'COMPLETED' ? '✅' :
                    status === 'ERROR' ? '❌' :
                    status === 'STARTED' ? '▶️' : '⏳';
        console.log(`${icon} ${msg.payload?.step_name || 'Step'}: ${msg.payload?.message || status}`);
        break;

      case 'agent_announcement':
        const actionIcon = msg.payload?.action === 'joined' ? '➕' :
                          msg.payload?.action === 'left' ? '➖' : '📊';
        console.log(`${actionIcon} Agent ${msg.payload?.agent_id} ${msg.payload?.action || 'update'}`);
        if (msg.payload?.status_message) {
          console.log(`   ${msg.payload.status_message}`);
        }
        break;

      case 'agent_output':
        console.log(`🤖 ${msg.payload?.agent_id || 'Agent'}: ${msg.payload?.content || JSON.stringify(msg.payload).substring(0, 200)}`);
        break;

      case 'ui_message':
        console.log(`💬 UI: ${msg.payload?.message || JSON.stringify(msg.payload)}`);
        if (msg.payload?.requires_response) {
          console.log('   ⏳ Waiting for your response...');
        }
        break;

      case 'system_message':
      case 'system_update':
        console.log(`📢 System: ${msg.payload?.message || JSON.stringify(msg.payload)}`);
        break;

      case 'osb_status':
        console.log(`🔬 OSB Status: ${msg.payload?.message || JSON.stringify(msg.payload)}`);
        break;

      case 'osb_partial':
        console.log(`🔬 OSB ${msg.payload?.agent}: ${msg.payload?.content || JSON.stringify(msg.payload).substring(0, 200)}`);
        break;

      case 'osb_complete':
        console.log(`✅ OSB Complete: ${msg.payload?.content || JSON.stringify(msg.payload).substring(0, 200)}`);
        if (msg.payload?.recommendations) {
          console.log('💡 Recommendations:');
          msg.payload.recommendations.forEach((rec: any, i: number) => {
            console.log(`   ${i + 1}. ${rec}`);
          });
        }
        break;

      default:
        console.log(`   Data: ${JSON.stringify(msg.payload).substring(0, 200)}`);
    }
  }

  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
      console.log(`\n📤 Sent: ${message.type}`);
    } else {
      console.error('❌ WebSocket not connected');
    }
  }

  runFlow(flowName: string, prompt?: string): void {
    const message: any = {
      type: 'run_flow',
      flow: flowName
    };
    if (prompt) {
      message.prompt = prompt;
    }
    this.send(message);
  }

  sendUserMessage(text: string): void {
    this.send({
      type: 'ui_message',
      payload: { text }
    });
  }

  sendManagerResponse(text: string): void {
    this.send({
      type: 'manager_response',
      payload: { text }
    });
  }
}

async function main() {
  const host = process.env.BUTTERMILK_HOST || 'localhost';
  const port = parseInt(process.env.BUTTERMILK_PORT || '8000');

  console.log('🧪 Buttermilk Flow Test Client');
  console.log('==============================\n');

  const client = new FlowClient(host, port);

  try {
    await client.connect();

    console.log('Available commands:');
    console.log('  /flow <name> [prompt]  - Run a flow');
    console.log('  /msg <text>           - Send user message');
    console.log('  /yes, /no             - Send manager response');
    console.log('  /exit                 - Exit');
    console.log('\nOr just type to send as manager response\n');

    // Demo: automatically run test flow
    console.log('🎬 Demo: Running test flow...');
    client.runFlow('test');

    // Interactive prompt
    const prompt = () => {
      rl.question('> ', (input) => {
        if (input.startsWith('/flow ')) {
          const parts = input.slice(6).split(' ');
          const flowName = parts[0];
          const flowPrompt = parts.slice(1).join(' ');
          client.runFlow(flowName, flowPrompt || undefined);
        } else if (input.startsWith('/msg ')) {
          client.sendUserMessage(input.slice(5));
        } else if (input === '/yes') {
          client.sendManagerResponse('yes');
        } else if (input === '/no') {
          client.sendManagerResponse('no');
        } else if (input === '/exit') {
          process.exit(0);
        } else if (input) {
          // Default: send as manager response
          client.sendManagerResponse(input);
        }

        prompt();
      });
    };

    // Start interactive prompt after a delay
    setTimeout(() => {
      console.log('\n💡 Try: /flow osb What is artificial intelligence?\n');
      prompt();
    }, 2000);

  } catch (error) {
    console.error('❌ Failed to connect:', error);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
