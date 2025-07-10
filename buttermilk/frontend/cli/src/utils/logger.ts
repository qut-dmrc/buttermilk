export class Logger {
  private static instance: Logger;
  private debugEnabled: boolean = false;
  private startTime: number = Date.now();

  private constructor() {}

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  setDebug(enabled: boolean): void {
    this.debugEnabled = enabled;
    if (enabled) {
      this.log('Debug logging enabled');
    }
  }

  private getTimestamp(): string {
    const elapsed = Date.now() - this.startTime;
    return `[+${(elapsed / 1000).toFixed(3)}s]`;
  }

  debug(...args: any[]): void {
    if (this.debugEnabled) {
      console.log(this.getTimestamp(), '[DEBUG]', ...args);
    }
  }

  log(...args: any[]): void {
    if (this.debugEnabled) {
      console.log(this.getTimestamp(), '[LOG]', ...args);
    }
  }

  error(...args: any[]): void {
    console.error(this.getTimestamp(), '[ERROR]', ...args);
  }

  ws(direction: 'send' | 'recv', message: any): void {
    if (this.debugEnabled) {
      const prefix = direction === 'send' ? '→ WS' : '← WS';
      console.log(this.getTimestamp(), prefix, JSON.stringify(message, null, 2));
    }
  }

  http(method: string, url: string, status?: number): void {
    if (this.debugEnabled) {
      const statusStr = status ? ` (${status})` : '';
      console.log(this.getTimestamp(), '[HTTP]', method, url + statusStr);
    }
  }
}

export const logger = Logger.getInstance();