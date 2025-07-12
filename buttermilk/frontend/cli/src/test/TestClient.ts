import { spawn, ChildProcess } from 'child_process';
import { Message } from '../types.js';

export interface TestClientOptions {
  cliPath: string;
  host?: string;
  port?: number;
  debug?: boolean;
}

export class TestClient {
  private process: ChildProcess | null = null;
  private output: string[] = [];
  private errorOutput: string[] = [];
  private connected: boolean = false;
  private onConnectedCallbacks: Array<() => void> = [];
  private onMessageCallbacks: Array<(message: string) => void> = [];

  constructor(private options: TestClientOptions) {}

  async start(): Promise<void> {
    const args = ['dist/cli.js'];
    if (this.options.host) {
      args.push('--host', this.options.host);
    }
    if (this.options.port) {
      args.push('--port', this.options.port.toString());
    }

    this.process = spawn('node', args, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        NODE_ENV: 'test',
        // Disable color output for easier parsing
        FORCE_COLOR: '0'
      }
    });

    this.process.stdout?.on('data', (data) => {
      const output = data.toString();
      this.output.push(output);
      
      if (this.options.debug) {
        console.log('[TestClient stdout]:', output);
      }

      // Check for connection
      if (output.includes('Connected') && !this.connected) {
        this.connected = true;
        this.onConnectedCallbacks.forEach(cb => cb());
        this.onConnectedCallbacks = [];
      }

      // Notify message callbacks
      this.onMessageCallbacks.forEach(cb => cb(output));
    });

    this.process.stderr?.on('data', (data) => {
      const error = data.toString();
      this.errorOutput.push(error);
      
      if (this.options.debug) {
        console.error('[TestClient stderr]:', error);
      }
    });

    this.process.on('error', (error) => {
      console.error('[TestClient] Process error:', error);
    });

    this.process.on('exit', (code) => {
      if (this.options.debug) {
        console.log(`[TestClient] Process exited with code ${code}`);
      }
    });
  }

  async waitForConnection(timeoutMs: number = 5000): Promise<void> {
    if (this.connected) return;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'));
      }, timeoutMs);

      this.onConnectedCallbacks.push(() => {
        clearTimeout(timeout);
        resolve();
      });
    });
  }

  async sendInput(text: string): Promise<void> {
    if (!this.process?.stdin) {
      throw new Error('Process not started or stdin not available');
    }

    return new Promise((resolve, reject) => {
      this.process!.stdin!.write(text + '\n', (error) => {
        if (error) {
          reject(error);
        } else {
          if (this.options.debug) {
            console.log('[TestClient] Sent input:', text);
          }
          resolve();
        }
      });
    });
  }

  async waitForMessage(
    matcher: string | RegExp | ((output: string) => boolean),
    timeoutMs: number = 5000
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Timeout waiting for message matching: ${matcher}`));
      }, timeoutMs);

      const checkExisting = () => {
        const allOutput = this.output.join('');
        if (this.matchesOutput(allOutput, matcher)) {
          clearTimeout(timeout);
          resolve(allOutput);
          return true;
        }
        return false;
      };

      // Check existing output first
      if (checkExisting()) return;

      // Wait for new output
      const callback = (output: string) => {
        if (this.matchesOutput(output, matcher)) {
          clearTimeout(timeout);
          this.onMessageCallbacks = this.onMessageCallbacks.filter(cb => cb !== callback);
          resolve(output);
        }
      };

      this.onMessageCallbacks.push(callback);
    });
  }

  private matchesOutput(
    output: string,
    matcher: string | RegExp | ((output: string) => boolean)
  ): boolean {
    if (typeof matcher === 'string') {
      return output.includes(matcher);
    } else if (matcher instanceof RegExp) {
      return matcher.test(output);
    } else {
      return matcher(output);
    }
  }

  async sendCommand(command: string): Promise<void> {
    await this.sendInput(command);
  }

  async runFlow(flowName: string, prompt?: string): Promise<void> {
    const command = prompt ? `/flow ${flowName} ${prompt}` : `/flow ${flowName}`;
    await this.sendCommand(command);
  }

  getOutput(): string[] {
    return [...this.output];
  }

  getErrorOutput(): string[] {
    return [...this.errorOutput];
  }

  getAllOutput(): string {
    return this.output.join('');
  }

  async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.process) {
        this.process.on('exit', () => {
          this.process = null;
          resolve();
        });
        this.process.kill();
      } else {
        resolve();
      }
    });
  }
}

// Test scenario runner
export interface TestScenario {
  name: string;
  steps: TestStep[];
  expectedOutcomes?: string[];
}

export interface TestStep {
  action: 'input' | 'wait' | 'waitFor' | 'assert';
  value?: string;
  timeout?: number;
  matcher?: string | RegExp;
}

export class ScenarioRunner {
  constructor(private client: TestClient) {}

  async runScenario(scenario: TestScenario): Promise<{
    success: boolean;
    errors: string[];
    output: string;
  }> {
    console.log(`[ScenarioRunner] Running scenario: ${scenario.name}`);
    const errors: string[] = [];

    try {
      await this.client.waitForConnection();

      for (const step of scenario.steps) {
        try {
          await this.executeStep(step);
        } catch (error) {
          errors.push(`Step failed: ${JSON.stringify(step)} - ${error}`);
        }
      }

      // Check expected outcomes
      if (scenario.expectedOutcomes) {
        const output = this.client.getAllOutput();
        for (const expected of scenario.expectedOutcomes) {
          if (!output.includes(expected)) {
            errors.push(`Expected outcome not found: "${expected}"`);
          }
        }
      }

      return {
        success: errors.length === 0,
        errors,
        output: this.client.getAllOutput()
      };
    } catch (error) {
      errors.push(`Scenario failed: ${error}`);
      return {
        success: false,
        errors,
        output: this.client.getAllOutput()
      };
    }
  }

  private async executeStep(step: TestStep): Promise<void> {
    switch (step.action) {
      case 'input':
        if (!step.value) throw new Error('Input step requires value');
        await this.client.sendInput(step.value);
        break;

      case 'wait':
        await new Promise(resolve => setTimeout(resolve, step.timeout || 1000));
        break;

      case 'waitFor':
        if (!step.matcher) throw new Error('WaitFor step requires matcher');
        await this.client.waitForMessage(step.matcher, step.timeout);
        break;

      case 'assert':
        if (!step.value) throw new Error('Assert step requires value');
        const output = this.client.getAllOutput();
        if (!output.includes(step.value)) {
          throw new Error(`Assertion failed: "${step.value}" not found in output`);
        }
        break;
    }
  }
}