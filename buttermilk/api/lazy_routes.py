"""Lazy route loading system for Phase 2 startup optimizations."""

import asyncio
from typing import Dict, List, Callable, Any
from fastapi import FastAPI, APIRouter
from buttermilk._core import logger


class LazyRouteManager:
    """Manages lazy loading of FastAPI routes and routers."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self._deferred_routers: List[Dict[str, Any]] = []
        self._core_routes_registered = False
        self._heavy_routes_registered = False
        
    def register_core_routes(self):
        """Register only essential routes for immediate functionality."""
        if self._core_routes_registered:
            return
            
        logger.info("Registering core routes only...")
        
        # Core health check route
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok", "message": "Core routes loaded"}
            
        self._core_routes_registered = True
        logger.info("Core routes registered successfully")
    
    def defer_router(self, router: APIRouter, prefix: str = "", **kwargs):
        """Defer registration of a heavy router until first request."""
        self._deferred_routers.append({
            "router": router,
            "prefix": prefix,
            "kwargs": kwargs
        })
        logger.info(f"Deferred router registration: {prefix}")
    
    async def load_heavy_routes_on_demand(self):
        """Load all deferred routes when first heavy request is made."""
        if self._heavy_routes_registered:
            return
            
        logger.info("Loading deferred routes on first heavy request...")
        
        for router_config in self._deferred_routers:
            try:
                self.app.include_router(
                    router_config["router"],
                    prefix=router_config["prefix"],
                    **router_config["kwargs"]
                )
                logger.info(f"Loaded deferred router: {router_config['prefix']}")
            except Exception as e:
                logger.error(f"Failed to load deferred router {router_config['prefix']}: {e}")
        
        self._heavy_routes_registered = True
        logger.info("All deferred routes loaded successfully")

    def create_lazy_middleware(self):
        """Create middleware that loads heavy routes on first request."""
        @self.app.middleware("http")
        async def lazy_route_loader(request, call_next):
            # Check if this is a request that needs heavy routes
            path = request.url.path
            if self._needs_heavy_routes(path) and not self._heavy_routes_registered:
                await self.load_heavy_routes_on_demand()
            
            response = await call_next(request)
            return response
    
    def _needs_heavy_routes(self, path: str) -> bool:
        """Determine if a request path requires heavy routes to be loaded."""
        heavy_route_prefixes = [
            "/api/flows",
            "/api/records", 
            "/api/session",
            "/tools/",
            "/ws/"
        ]
        return any(path.startswith(prefix) for prefix in heavy_route_prefixes)


def create_core_router() -> APIRouter:
    """Create router with only essential routes that need to be loaded immediately."""
    router = APIRouter()
    
    from fastapi import Request, HTTPException
    from fastapi.responses import StreamingResponse
    from buttermilk._core.types import RunRequest
    
    @router.api_route("/flow/{flow_name}", methods=["GET", "POST"])
    async def run_flow_json(
        flow_name: str,
        request: Request,
        run_request: RunRequest | None = None,
    ) -> StreamingResponse:
        """Run a flow with provided inputs - core functionality."""
        # Access state via request.app.state
        if not hasattr(request.app.state.flow_runner, "flows") or flow_name not in request.app.state.flow_runner.flows:
            raise HTTPException(status_code=404, detail="Flow configuration not found or flow name invalid")

        # Use stream method with the flow runner
        return StreamingResponse(
            request.app.state.flow_runner.stream(run_request),
            media_type="application/json",
        )
    
    return router