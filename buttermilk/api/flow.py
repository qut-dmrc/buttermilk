import asyncio
from collections.abc import AsyncGenerator, Sequence
import uuid 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import pydantic
import uvicorn
import asyncio
import random
from buttermilk._core.contract import ErrorEvent, ManagerResponse, RunRequest
from buttermilk.bm import BM, logger
from buttermilk.runner.flowrunner import FlowRunner
import json
import logging
import os
from typing import Any, Awaitable, Callable, Optional

import aiofiles
import yaml
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .runs import get_recent_runs

INPUT_SOURCE = "api"

def create_app(bm: BM, flows: FlowRunner) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI()
    
    # Set up state
    app.state.bm = bm
    app.state.flow_runner = flows

    # curl -X 'POST' 'http://127.0.0.1:8000/flow/simple' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"q": "Democrats are arseholes."}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/simple' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text": "Democrats are arseholes."}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/trans' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"uri": "https://www.city-journal.org/article/what-are-we-doing-to-children"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/osb' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"q": "Is it still hate speech if the targeted group is not explicitly named?"}'

    # curl -X 'POST' 'http://127.0.0.1:8000/flow/hate' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "criteria": "criteria_ordinary", "video": "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/hate' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"uri": "https://upload.wikimedia.org/wikipedia/en/b/b9/MagrittePipe.jpg"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "template":"summarise_osb", "text": "gs://dmrc-platforms/data/osb/FB-UK2RUS24.md"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/trans' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"record_id": "betoota_snape_trans"}'


    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )


    @app.api_route("/flow/{flow_name}", methods=["GET", "POST"])
    async def run_flow_json(
        flow_name: str,
        request: Request,
        run_request: RunRequest | None = "",
    ) -> StreamingResponse:
        """Run a flow with provided inputs."""

        # Access state via request.app.state
        if not hasattr(request.app.state.flow_runner, "flows") or flow_name not in request.app.flow_runner.flows:
            raise HTTPException(status_code=404, detail="Flow configuration not found or flow name invalid")
        
        # Use flow_stream with the new flow_runner
        return StreamingResponse(
            request.app.flow_runner.stream(run_request),
            media_type="application/json",
        )

    @app.api_route("/html/flow/{flow}", methods=["GET", "POST"])
    @app.api_route("/html/flow", methods=["GET", "POST"])
    async def run_route_html(
        request: Request,
        flow: str = "",
        flow_request: RunRequest | None = "",
    ) -> StreamingResponse:
        if flow not in flow:
            raise HTTPException(status_code=403, detail="Flow not valid")

        async def result_generator() -> AsyncGenerator[str, None]:
            logger.debug(
                f"Received request for HTML flow {flow} and flow_request {flow_request}",
            )
            try:
                async for data in flow_stream(
                    flows[flow],
                    flow_request=flow_request,
                    return_json=False,
                ):
                    # Render the template with the response data
                    rendered_result = templates.TemplateResponse(
                        "flow_html.html",
                        {"request": request, "data": data},
                    )
                    yield rendered_result.body.decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return StreamingResponse(result_generator(), media_type="text/html")


    # Set up CORS
    origins = [
        "http://localhost:5000",  # Frontend running on localhost:5000
        "http://127.0.0.1:5000",
        "http://127.0.0.1:8080",  # Frontend running on localhost:8080
        "http://localhost:8080",
        "http://localhost:8000",  # Allow requests from localhost:8000
        "http://127.0.0.1:8000",
        "http://automod.cc",  # Allow requests from your domain
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow specific origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"],  # Allow all headers
    )


    # Custom middleware to log CORS failures
    @app.middleware("http")
    async def log_cors_failures(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin:
            logger.debug(f"CORS check for {origin}")

        response = await call_next(request)
        return response

    @app.websocket("/ws/{session_id}")
    async def agent_websocket(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for client communication with the WebUIAgent.
        
        Args:
            websocket: WebSocket connection
            session_id: Unique identifier for this client session
        """

        # Accept the connection first
        await websocket.accept()
        client_listener_task = None
        try:
            # Start the flow, passing in our websocket and session_id
            # Wait for client message
            while True:
                client_message = await websocket.receive_json()
                try:
                    run_request = RunRequest.model_validate(client_message)
                    run_request.client_callback=websocket
                    run_request.session_id=session_id
                    break
                except (pydantic.ValidationError, json.JSONDecodeError):
                    await websocket.send_json(ErrorEvent(source="fastapi flow websocket", content="Send a valid RunRequest to start."))
                
            agent_callback = await app.state.flow_runner.run_flow(run_request)
            
            async def listen_client():
                """Task to listen for incoming messages from the client."""
                while True:
                    try:
                        incoming_data = await websocket.receive_json()
                        # Forward data to the running flow via the handler
                        await agent_callback(incoming_data) 
                    except WebSocketDisconnect:
                        logger.info(f"Client {session_id} disconnected.")
                        # Optionally signal the flow task to stop if needed
                        # if flow_task_handle:
                        #     flow_task_handle.cancel()
                        break # Exit loop on disconnect
                    except Exception as e:
                        logger.error(f"Error receiving/processing client message for {session_id}: {e}")
                        # Decide if you want to break or continue
                        # break # Exit loop on other errors

            client_listener_task = asyncio.create_task(listen_client())

            await client_listener_task

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Unexpected error: {str(e)}",
                    "source": "system"
                })
            except:
                pass
        finally:
            if client_listener_task and not client_listener_task.done():
                client_listener_task.cancel()

    
    # Helper route to generate session IDs for clients
    @app.get("/api/session")
    async def create_session():
        """
        Generates a unique session ID for new web clients.
        
        Returns:
            Dict with new session ID
        """
        return {"session_id": str(uuid.uuid4())}
    




    # --- Import Shiny App object ---
    from buttermilk.web.shiny import get_shiny_app

    # --- Mount the Shiny App ---
    shiny_app_asgi = get_shiny_app(flows=flows)
    app.mount("/ui", shiny_app_asgi, name="shiny_app")

    return app

