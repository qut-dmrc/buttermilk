import asyncio
from collections.abc import Mapping
import importlib
from typing import AsyncGenerator, Literal, Optional, Dict, Any, Union, Type

from pydantic import BaseModel, ConfigDict, Field

from buttermilk._core.config import SaveInfo
from buttermilk._core.contract import RunRequest
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk.bm import BM, LLMs, logger

class FlowRunner(BaseModel):
    """
    Centralized service for running flows across different entry points.
    
    Handles orchestrator instantiation and execution in a consistent way, regardless
    of whether the flow is started from CLI, API, Slackbot, or Pub/Sub.
    """
    bm: BM
    flows: Mapping[str, OrchestratorProtocol]  # Dictionary of unconfigured flow orchestrators.
    save: SaveInfo
    tasks: list = Field(default=[])
    ui: str
    model_config = ConfigDict(extra="allow")
    
    async def run_flow(self, 
                        run_request: RunRequest,
                      **kwargs) -> Any:
        """
        Run a flow based on its configuration and a request.
        
        Args:
            run_request: The request containing input parameters
            history: Optional conversation history (for chat-based interfaces)
            **kwargs: Additional keyword arguments for orchestrator instantiation
            
        Returns:
            Task object if running asynchronously in background, otherwise None
        
        Raises:
            ValueError: If orchestrator isn't specified or unknown
        """
           
        module_name, class_name = self.flows[run_request.flow].orchestrator.rsplit(".", 1)
        module = importlib.import_module(module_name)
        orchestrator_cls = getattr(module, class_name)
    
        config = self.flows[run_request.flow].model_dump()
        # Create orchestrator instance
        orchestrator = orchestrator_cls(**config)
        
        callback = orchestrator._make_publish_callback()

        logger.info(f"Starting flow '{run_request.flow}' with orchestrator '{orchestrator_cls.__name__}'")
        self.tasks.append(asyncio.create_task(orchestrator.run(request=run_request)))
        
        return callback

