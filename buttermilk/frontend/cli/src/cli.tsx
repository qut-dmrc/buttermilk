#!/usr/bin/env node
import React, { useState, useEffect } from 'react';
import { render, Text } from 'ink';
import meow from 'meow';
import UI from './ui.js';
import UIReadline from './ui-readline.js';
import http from 'http';
import https from 'https';
import Spinner from 'ink-spinner';
import { logger } from './utils/logger.js';

const cli = meow(`
  Usage
    $ buttermilk-cli

  Options
    --host, -h      Backend host (default: localhost)
    --port, -p      Backend port (default: 8080)
    --url, -u       Full backend URL (overrides host/port)
    --debug, -d     Enable debug logging

  Environment Variables
    BUTTERMILK_HOST    Backend host
    BUTTERMILK_PORT    Backend port
    BUTTERMILK_URL     Full backend URL
    BUTTERMILK_DEBUG   Enable debug logging

  Description
    Connects to the Buttermilk backend via WebSocket.
`,
{
	importMeta: import.meta,
	flags: {
		host: {
			type: 'string',
			shortFlag: 'h',
			default: process.env.BUTTERMILK_HOST || 'localhost'
		},
		port: {
			type: 'number',
			shortFlag: 'p',
			default: parseInt(process.env.BUTTERMILK_PORT || '8080')
		},
		url: {
			type: 'string',
			shortFlag: 'u',
			default: process.env.BUTTERMILK_URL || ''
		},
		debug: {
			type: 'boolean',
			shortFlag: 'd',
			default: process.env.BUTTERMILK_DEBUG === 'true'
		},
		readline: {
			type: 'boolean',
			shortFlag: 'r',
			default: process.env.BUTTERMILK_READLINE === 'true' || true
		}
	}
});

// Enable debug logging if requested
logger.setDebug(cli.flags.debug);

const BACKEND_HOST = cli.flags.url && cli.flags.url !== '' ? new URL(cli.flags.url).hostname : cli.flags.host;
const BACKEND_PORT = cli.flags.url && cli.flags.url !== '' ? parseInt(new URL(cli.flags.url).port || '80') : cli.flags.port;
const BACKEND_PROTOCOL = cli.flags.url && cli.flags.url !== '' ? new URL(cli.flags.url).protocol.replace(':', '') : 'http';

logger.debug('CLI Configuration:', {
  host: BACKEND_HOST,
  port: BACKEND_PORT,
  protocol: BACKEND_PROTOCOL,
  debug: cli.flags.debug
});

const App = () => {
  const [websocketUrl, setWebsocketUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSessionId = async () => {
      console.log("App: Starting fetchSessionId...");
      try {
        const response = await new Promise<string>((resolve, reject) => {
          const httpModule = BACKEND_PROTOCOL === 'https' ? https : http;
          httpModule.get(`${BACKEND_PROTOCOL}://${BACKEND_HOST}:${BACKEND_PORT}/api/session`, (res) => {
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
        const wsProtocol = BACKEND_PROTOCOL === 'https' ? 'wss' : 'ws';
        const newWebsocketUrl = `${wsProtocol}://${BACKEND_HOST}:${BACKEND_PORT}/ws/${sessionId}`;
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
  // Use readline-based UI by default for proper input echo
  return cli.flags.readline ? <UIReadline url={websocketUrl} /> : <UI url={websocketUrl} />;
};

render(<App />);