# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Chat Frontend Development

This is a SvelteKit frontend for the Buttermilk project, implementing a retro terminal-style chat interface styled after classic IRC clients. The application serves as a research tool for HASS (Humanities, Arts, Social Sciences) academics working with AI assessment systems.

## Development Commands

- **Install dependencies**: `npm install`
- **Development server**: `npm run dev` (runs with --host and info logging)
- **Build**: `npm run build`
- **Preview production**: `npm run preview` (builds and runs with Wrangler)
- **Type checking**: `npm run check` (or `npm run check:watch` for watch mode)
- **Linting**: `npm run lint` (combines Prettier and ESLint checks)
- **Format code**: `npm run format`
- **Deploy to Cloudflare**: `npm run deploy`
- **Generate Cloudflare types**: `npm run cf-typegen`

## Architecture Overview

### Core Design Philosophy
- **Retro Terminal Group Chat**: IRC-style interface with terminal aesthetics
- **Academic Research Tool**: Built for HASS researchers analyzing AI assessment systems
- **Minimal Dependencies**: Uses Bootstrap, avoids custom CSS where possible
- **Real-time Communication**: WebSocket-based message streaming

### Key Technologies
- **SvelteKit**: Main framework with SSR and API routes
- **Bootstrap 5**: UI framework with retro terminal theming
- **HTMX**: Enhanced HTML interactions for real-time updates
- **WebSockets**: Real-time message communication
- **MDSvex**: Markdown processing in Svelte components
- **Cloudflare**: Deployment target using Workers/Pages

### Project Structure
```
src/
├── lib/
│   ├── components/           # Reusable UI components
│   │   ├── layout/          # Header, Nav, Sidebar components
│   │   └── messages/        # Message type components (Judge, Researcher, etc.)
│   ├── stores/              # Svelte stores for state management
│   └── utils/               # Utility functions
├── routes/                  # SvelteKit routes
│   ├── api/                # Backend API proxy endpoints
│   └── terminal/           # Terminal-specific pages
└── types/                  # TypeScript definitions
```

### State Management
- **apiStore.ts**: Reactive stores for API data with caching
- **messageStore.ts**: Chat message state
- **sessionStore.ts**: Session management
- **terminalActionsStore.ts**: Terminal UI interactions

### API Integration
The frontend uses a proxy pattern for backend communication:
- Frontend API routes (`/api/*`) proxy to backend services
- WebSocket connections for real-time message streaming
- Fallback mock data for development when backend unavailable
- Session-based WebSocket connections via `/api/session`

### Message System
Supports multiple agent message types:
- **AssessmentMessage**: AI assessment results
- **JudgeMessage**: Judge evaluations with model/criteria info
- **ResearcherMessage**: Research analysis
- **SummaryMessage**: Compact summary views
- **Basic/Record/Differences**: Other message types

### Configuration
- **Environment**: Uses `.env` for backend API URL configuration
- **Vite**: Configured with CORS, WebSocket proxy, and module optimization
- **TypeScript**: Strict mode with Cloudflare Workers types
- **Wrangler**: Cloudflare deployment configuration in `wrangler.jsonc`

## Development Guidelines

### Code Style
- Follow existing Bootstrap patterns over custom CSS
- Use TypeScript for all new code
- Implement proper error handling for API calls
- Use reactive stores for state management
- Follow SvelteKit conventions for routing and components

### Theme Implementation
- **Retro Terminal**: IRC-style interface with terminal colors
- **Two Themes**: Classic retro and modern terminal variants in `_theme-*.scss`
- **Agent Representation**: Color-coded by model, avoid emoji/cartoonish icons
- **Compact UI**: Minimize whitespace, focus on information density

### Integration Points
- **Backend API**: Configured via `BACKEND_API_URL` environment variable
- **WebSocket**: Real-time communication for chat messages
- **Session Management**: Session-based connections for multi-user support
- **Message Routing**: Type-based message component rendering

### Testing & Quality
- Use `npm run check` for TypeScript validation
- Run `npm run lint` for code quality checks
- Test WebSocket connections via `/api/websocket` endpoint
- Validate API proxy functionality through `/api/*` routes