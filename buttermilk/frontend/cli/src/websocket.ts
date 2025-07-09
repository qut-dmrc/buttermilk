
import WebSocket from 'ws';
import { Message } from './types.js';

export const connect = (url: string, onMessage: (message: Message) => void) => {
  console.log(`WebSocket: Attempting to connect to WebSocket: ${url}`);
  const ws = new WebSocket(url);

  ws.on('open', () => {
    console.log('WebSocket: Connected to server');
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
    console.log(`WebSocket: Disconnected from server. Code: ${code}, Reason: ${reason.toString()}`);
  });

  ws.on('error', (error) => {
    console.error('WebSocket: Error:', error);
  });

  return {
    send: (message: Message) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      } else {
        console.warn('WebSocket: Cannot send message, connection not open.');
      }
    },
    close: () => {
      ws.close();
    }
  };
};

