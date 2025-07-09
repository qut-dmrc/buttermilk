#!/usr/bin/env node
import React, { useState, useEffect } from 'react';
import { render, Text } from 'ink';
import meow from 'meow';
import UI from './ui.js';
import http from 'http'; // Import http module
import Spinner from 'ink-spinner'; // Assuming ink-spinner is available

const BACKEND_HOST = 'localhost';
const BACKEND_PORT = 8080;

const cli = meow(`
  Usage
    $ buttermilk-cli

  Description
    Connects to the Buttermilk backend via WebSocket.
`,
{
	importMeta: import.meta,
});

const App = () => {
  const [websocketUrl, setWebsocketUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSessionId = async () => {
      console.log("App: Starting fetchSessionId...");
      try {
        const response = await new Promise<string>((resolve, reject) => {
          http.get(`http://${BACKEND_HOST}:${BACKEND_PORT}/api/session`, (res) => {
            let data = '';
            res.on('data', (chunk) => {
              data += chunk;
            });
            res.on('end', () => {
              resolve(data);
            });
          }).on('error', (err) => {
            reject(err);
          });
        });

        const sessionData = JSON.parse(response);
        const sessionId = sessionData.session_id;
        if (!sessionId) {
          throw new Error('Session ID not found in response');
        }
        const newWebsocketUrl = `ws://${BACKEND_HOST}:${BACKEND_PORT}/ws/${sessionId}`;
        console.log(`App: Fetched session ID, setting WebSocket URL: ${newWebsocketUrl}`);
        setWebsocketUrl(newWebsocketUrl);
      } catch (err: any) {
        console.error(`App: Error in fetchSessionId: ${err.message}`);
        setError(`Failed to get session ID: ${err.message}`);
      }
    };

    fetchSessionId();
  }, []);

  if (error) {
    console.log(`App: Rendering error: ${error}`);
    return <Text color="red">{error}</Text>;
  }

  if (!websocketUrl) {
    console.log("App: Rendering spinner (fetching session ID)...");
    return (
      <Text>
        <Spinner type="dots" /> Fetching session ID...
      </Text>
    );
  }

  console.log(`App: Rendering UI with URL: ${websocketUrl}`);
  return <UI url={websocketUrl} />;
};

render(<App />, {
  stdin: undefined,
});

render(<App />, {
  stdin: undefined,
});