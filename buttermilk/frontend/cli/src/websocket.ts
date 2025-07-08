
import WebSocket from 'ws';
import { Message } from './types.js';

export const connect = (url: string, onMessage: (message: Message) => void) => {
  const ws = new WebSocket(url);

  ws.on('open', () => {
    console.log('Connected to websocket server');
  });

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data.toString());
      onMessage(message);
    } catch (error) {
      console.error('Error parsing message:', error);
    }
  });

  ws.on('close', () => {
    console.log('Disconnected from websocket server');
  });

  ws.on('error', (error) => {
    console.error('Websocket error:', error);
  });

  return {
    send: (message: Message) => {
      ws.send(JSON.stringify(message));
    },
    close: () => {
      ws.close();
    }
  };
};
