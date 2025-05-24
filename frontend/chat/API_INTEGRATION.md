# API Integration Guide

This document explains how the Svelte frontend connects to your backend API.

## Configuration

The application uses a `.env` file in the `chat/` directory to configure the backend URL:

```
# Backend API URL
BACKEND_API_URL="http://localhost:8080"
```

Update this URL to point to your actual backend server. For production deployment, set this environment variable appropriately.

## How It Works

### API Proxy 

The frontend uses a proxy approach for API requests. Here's how it works:

1. Frontend makes a request to its own endpoint (e.g., `/api/flows`)
2. The SvelteKit endpoint proxies this request to the backend (e.g., `http://localhost:8080/api/flows`)
3. The backend response is returned to the frontend

This approach offers several benefits:
- Avoids CORS issues
- Provides fallback data when the backend is unavailable
- Allows for centralized error handling
- Simplifies authentication by handling it server-side

### API Endpoints

The following endpoints are used:

| Frontend Endpoint | Backend Endpoint | Description |
|-------------------|------------------|-------------|
| `/api/flows` | `${BACKEND_API_URL}/api/flows` | Gets list of available flows |
| `/api/records?flow=X` | `${BACKEND_API_URL}/api/records?flow=X` | Gets records for a specific flow |
| `/api/session` | `${BACKEND_API_URL}/api/session` | Gets a session ID for WebSocket connection |
| `/ws/${sessionId}` | `${BACKEND_API_URL}/ws/${sessionId}` | WebSocket connection |

### Data Flow

1. The `apiStore.ts` provides reactive stores that components can subscribe to
2. The ApiDropdown component connects to these stores and displays their data
3. When a flow is selected, the records store automatically updates to fetch filtered records

## Error Handling

The system includes robust error handling:

1. API requests that fail will show error messages in the UI
2. If the backend is unavailable during development, fallback mock data is provided
3. Loading states are tracked and displayed to users

## Extending the System

### Adding New Endpoints

To add a new API endpoint:

1. Create a new file in `chat/src/routes/api/your-endpoint/+server.ts`
2. Use the proxy pattern to forward requests to the backend
3. Create a new store in `apiStore.ts` for your endpoint

### Adding New Components

The ApiDropdown component can be used with any API store. To create a new dropdown:

```svelte
<ApiDropdown 
  store={yourStore}
  label="Your Label"
  placeholder="Choose an option..."
  bind:value={selectedValue}
  valueProperty="id"  <!-- Optional: if using objects -->
  labelProperty="name"  <!-- Optional: if using objects -->
/>
```

## Authentication

If your API requires authentication:

1. Add appropriate headers to the API proxy endpoints
2. Set up token storage and refresh logic as needed
3. Configure the frontend to include authentication information with requests
