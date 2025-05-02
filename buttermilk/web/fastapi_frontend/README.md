# Buttermilk FastAPI Frontend

This directory contains a modular FastAPI frontend for the Buttermilk project. The code has been reorganized to improve maintainability and separation of concerns.

## Structure

- **app.py**: Main application entry point that creates and configures the FastAPI app.
- **routes.py**: API route definitions for the dashboard.
- **schemas.py**: Pydantic models for data validation and serialization.
- **services/**: Service layer that contains business logic.
  - **data_service.py**: Handles data access and processing for records, criteria, etc.
  - **message_service.py**: Handles message formatting and extraction.
  - **ui_service.py**: UI utility functions for formatting and display.
  - **websocket_service.py**: WebSocket connection management and message handling.
- **templates/**: Jinja2 HTML templates.
  - **base.html**: Base template with common layout.
  - **index.html**: Main dashboard page.
  - **dashboard.html**: Dashboard reports page.
  - **partials/**: Partial templates used with HTMX for dynamic content.

## Technology Stack

- **FastAPI**: High-performance web framework
- **Jinja2**: Template engine
- **HTMX**: Frontend interactions without heavy JavaScript
- **Alpine.js**: Lightweight JavaScript framework for declarative UI
- **Tailwind CSS**: Utility-first CSS framework

## Design Principles

1. **Separation of Concerns**: Each component has a single responsibility.
2. **Modularity**: Components are designed to be reusable and independently testable.
3. **Progressive Enhancement**: Core functionality works without JavaScript, with enhancements when available.
4. **Type Safety**: Extensive use of type hints and Pydantic models.

## Usage

The frontend is initialized through the `create_dashboard_app` function in `app.py`:

```python
from buttermilk.web.fastapi_frontend.app import create_dashboard_app

# Create a FastAPI app with the dashboard
app = create_dashboard_app(flows=flow_runner)
```

## Frontend JavaScript Architecture

The frontend uses a hybrid approach:

1. **HTMX** for server-rendered partial updates
2. **Alpine.js** for client-side state management
3. **WebSockets** for real-time communication

### Data Flow

1. User interactions trigger HTMX requests to the server
2. FastAPI routes handle requests and return HTML fragments
3. WebSocket connections maintain real-time state updates
4. Alpine.js manages local UI state and user interactions

## Extending the Frontend

### Adding a New API Endpoint

1. Add a new route function in `routes.py`
2. If needed, add a new template in `templates/`
3. For data processing, add a method to the appropriate service

### Adding a New WebSocket Message Type

1. Add a new model in `schemas.py`
2. Add a handler method in `websocket_service.py`
3. Update the client-side code in `index.html` or relevant template

### Modifying UI Components

The UI follows a component-based approach with Tailwind CSS:

1. Common UI patterns are in `ui_service.py`
2. Reusable HTML fragments are in `templates/partials/`
3. Alpine.js components handle interactive behaviors

## Potential Future Improvements

1. **Component Libraries**: Consider using TailwindUI, DaisyUI, or HeadlessUI for more consistent components.
2. **Client-Side Caching**: Implement caching for API responses to reduce server load.
3. **Server-Sent Events**: For some one-way communications, SSE may be more efficient than WebSockets.
4. **TypeScript**: Adding TypeScript for frontend code could improve type safety.
5. **Import Maps**: Use browser import maps for better JavaScript module organization.
6. **API Client Library**: Generate a client library from OpenAPI schema for frontend use.
