// Retro IRC-style theme configuration
// Based on the web frontend's terminal.scss theme

// Define colors separately to avoid circular references
const colors = {
  background: '#0D0D0D',
  text: '#00dd00',           // Lime green
  textDim: '#00aa00',        // Darker green for timestamps
  textBright: '#00ff00',     // Brighter green for hover/active
  
  // IRC nick colors (cycling through for different agents)
  nicks: [
    '#00dd00', // Default lime green
    '#00ffff', // Cyan
    '#ff9900', // Orange
    '#ff00ff', // Magenta
    '#ffff00', // Yellow
    '#00ff00', // Bright green
    '#9999ff', // Light blue
    '#ff6666', // Light red
  ],
  
  // Message type colors
  system: '#888888',
  error: '#cc0000',
  success: '#00cc00',
  warning: '#ff9900',
  info: '#00ffff',
  
  // Status colors
  connected: '#00cc00',
  disconnected: '#cc0000',
  reconnecting: '#ff9900',
  
  // UI elements
  border: '#333333',
  scrollbar: '#008800',
  input: {
    background: 'transparent',
    text: '#00dd00',
    placeholder: '#006600'
  }
};

// Agent color mapping
const agentColorMap = new Map([
  ['gpt-4', 0],
  ['gpt-3.5-turbo', 1],
  ['claude', 2],
  ['claude-instant', 3],
  ['user', 4],
  ['system', 5],
  ['judge', 6],
  ['researcher', 7],
]);

// Get color for agent
const getAgentColor = (agentId: string): string => {
  const lowerAgent = agentId.toLowerCase();
  for (const [key, colorIndex] of agentColorMap.entries()) {
    if (lowerAgent.includes(key)) {
      return colors.nicks[colorIndex];
    }
  }
  // Default: hash agent name to pick a color
  const hash = agentId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return colors.nicks[hash % colors.nicks.length];
};

export const retroIRCTheme = {
  // Base colors
  colors,
  
  // Layout constants
  layout: {
    nickWidth: 15,        // Characters width for nickname column
    nickSeparator: '│',   // Vertical bar separator
    timestampFormat: 'HH:mm:ss',
    messageIndent: 2,     // Spaces after separator
    maxNickLength: 14,    // Leave room for separator
  },
  
  // Agent display configuration
  agents: {
    // Map agent types/models to consistent colors
    colorMap: agentColorMap,
    // Get color for agent
    getColor: getAgentColor
  },
  
  // Format helpers
  format: {
    // Format nickname with padding and alignment
    nickname(nick: string, maxWidth: number = 14): string {
      if (nick.length > maxWidth) {
        return nick.substring(0, maxWidth - 1) + '…';
      }
      return nick.padStart(maxWidth);
    },
    
    // Format timestamp
    timestamp(date: Date | string): string {
      const d = typeof date === 'string' ? new Date(date) : date;
      const hours = d.getHours().toString().padStart(2, '0');
      const mins = d.getMinutes().toString().padStart(2, '0');
      const secs = d.getSeconds().toString().padStart(2, '0');
      return `${hours}:${mins}:${secs}`;
    },
    
    // Format button text
    button(text: string): string {
      return `[ ${text} ]`;
    },
    
    // Format status indicator
    status(status: 'connected' | 'disconnected' | 'reconnecting'): string {
      switch (status) {
        case 'connected':
          return '●';
        case 'disconnected':
          return '○';
        case 'reconnecting':
          return '◐';
      }
    }
  }
};

// Export type for theme
export type Theme = typeof retroIRCTheme;