import json
from collections.abc import AsyncGenerator
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from buttermilk.api import stream

from .testdata.jobs_summarise import jobs

app = FastAPI()
flows = ["summarise_osb"]

templates = Jinja2Templates(directory="buttermilk/api/templates")


async def mock_flow_stream(
    output_json=True,
    **kwargs,
) -> AsyncGenerator[str, None]:
    for data in jobs:
        if output_json:
            yield json.dumps(data)
        else:
            yield data


@app.api_route("/flow/{flow}", methods=["GET", "POST"])
async def run_flow_json(
    flow: Literal["hate", "trans", "osb", "osbfulltext", "summarise_osb"],
    request: Request,
    flow_request: stream.FlowRequest | None = "",
) -> StreamingResponse:
    if flow not in flows:
        raise HTTPException(status_code=403, detail=f"TEST: Flow {flow} not valid")

    return StreamingResponse(
        mock_flow_stream(),
        media_type="application/json",
    )


@app.api_route("/html/flow/{flow}", methods=["GET", "POST"])
@app.api_route("/html/flow", methods=["GET", "POST"])
async def run_route_html(
    request: Request,
    flow: str = "",
    flow_request: Any = None,
) -> StreamingResponse:
    if flow not in flows:
        raise HTTPException(status_code=403, detail="Flow not valid")

    async def result_generator() -> AsyncGenerator[str, None]:
        async for data in mock_flow_stream(output_json=False):
            # Render the template with the response data
            rendered_result = templates.TemplateResponse(
                "flow_html.html",
                {"request": request, "data": data},
            )
            yield rendered_result.body.decode("utf-8")

    return StreamingResponse(result_generator(), media_type="text/html")


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
        print(f"CORS check for {origin}")

    response = await call_next(request)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
