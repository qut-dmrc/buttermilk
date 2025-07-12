import WebSocket from 'ws';
import { Message } from './types.js';

export interface WebSocketConnection {
  send: (message: Message) => void;
  close: () => void;
  getState: () => ConnectionState;
}

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting' | 'error';

export type ConnectionCallback = (state: ConnectionState, error?: Error) => void;

interface ReconnectOptions {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  factor?: number;
}

const DEFAULT_RECONNECT_OPTIONS: Required<ReconnectOptions> = {
  maxRetries: 10,
  initialDelay: 1000,
  maxDelay: 30000,
  factor: 1.5,
};

export const connect = (
  url: string,
  onMessage: (message: Message) => void,
  onConnectionChange?: ConnectionCallback,
  reconnectOptions?: ReconnectOptions
): WebSocketConnection => {
  const options = { ...DEFAULT_RECONNECT_OPTIONS, ...reconnectOptions };
  
  let ws: WebSocket | null = null;
  let state: ConnectionState = 'connecting';
  let retryCount = 0;
  let retryDelay = options.initialDelay;
  let reconnectTimeout: NodeJS.Timeout | null = null;
  let shouldReconnect = true;
  let messageQueue: Message[] = [];

  const setState = (newState: ConnectionState, error?: Error) => {
    state = newState;
    onConnectionChange?.(newState, error);
  };

  const cleanup = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
    if (ws) {
      ws.removeAllListeners();
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
      ws = null;
    }
  };

  const processMessageQueue = () => {
    while (messageQueue.length > 0 && ws && ws.readyState === WebSocket.OPEN) {
      const message = messageQueue.shift()!;
      ws.send(JSON.stringify(message));
    }
  };

  const attemptConnection = () => {
    cleanup();
    
    if (!shouldReconnect) {
      return;
    }

    console.log(`WebSocket: Attempting to connect to: ${url}`);
    setState('connecting');
    
    ws = new WebSocket(url);

    ws.on('open', () => {
      console.log('WebSocket: Connected to server');
      setState('connected');
      retryCount = 0;
      retryDelay = options.initialDelay;
      processMessageQueue();
    });

    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        onMessage(message);
      } catch (error) {
        console.error('WebSocket: Error parsing message:', error);
      }
    });

    ws.on('close', (code, reason) => {
      console.log(`WebSocket: Disconnected. Code: ${code}, Reason: ${reason?.toString() || 'Unknown'}`);
      
      if (shouldReconnect && retryCount < options.maxRetries) {
        setState('reconnecting');
        retryCount++;
        
        console.log(`WebSocket: Reconnecting in ${retryDelay}ms (attempt ${retryCount}/${options.maxRetries})`);
        
        reconnectTimeout = setTimeout(() => {
          attemptConnection();
        }, retryDelay);
        
        retryDelay = Math.min(retryDelay * options.factor, options.maxDelay);
      } else {
        setState('disconnected');
        if (retryCount >= options.maxRetries) {
          console.error('WebSocket: Max reconnection attempts reached');
        }
      }
    });

    ws.on('error', (error) => {
      console.error('WebSocket: Error:', error);
      setState('error', error as Error);
    });
  };

  // Start initial connection
  attemptConnection();

  return {
    send: (message: Message) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      } else {
        console.warn('WebSocket: Queueing message, connection not open');
        messageQueue.push(message);
      }
    },
    close: () => {
      shouldReconnect = false;
      cleanup();
      setState('disconnected');
    },
    getState: () => state
  };
};