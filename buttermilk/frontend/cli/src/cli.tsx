#!/usr/bin/env node
import React from 'react';
import { render } from 'ink';
import meow from 'meow';
import UI from './ui.js';

const cli = meow(`
  Usage
    $ buttermilk-cli <websocket-url>
`);

if (cli.input.length === 0) {
  console.error('Please provide a websocket URL');
  process.exit(1);
}

render(<UI url={cli.input[0]} />);