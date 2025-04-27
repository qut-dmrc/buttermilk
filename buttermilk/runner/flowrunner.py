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

    model_config = ConfigDict(extra="allow")
    
    async def run_flow(self, 
                        flow_name: str,
                        run_request: Optional[RunRequest] = None,
                      history: Optional[list] = None,
                      **kwargs) -> Any:
        """
        Run a flow based on its configuration and a request.
        
        Args:
            flow_name: The configured flow name
            run_request: The request containing input parameters
            history: Optional conversation history (for chat-based interfaces)
            **kwargs: Additional keyword arguments for orchestrator instantiation
            
        Returns:
            Task object if running asynchronously in background, otherwise None
        
        Raises:
            ValueError: If orchestrator isn't specified or unknown
        """
        if not flow_name:
            logger.error("Orchestrator not specified in flow config")
            raise ValueError("Orchestrator not specified in flow config")
           
        module_name, class_name = self.flows[flow_name].orchestrator.rsplit(".", 1)
        module = importlib.import_module(module_name)
        orchestrator_cls = getattr(module, class_name)
    
        config = self.flows[flow_name].model_dump()
        # Create orchestrator instance
        orchestrator = orchestrator_cls(**config)
        
        callback = orchestrator._make_publish_callback()

        logger.info(f"Starting flow '{flow_name}' with orchestrator '{orchestrator_cls.__name__}'")
        self.tasks.append(asyncio.create_task(orchestrator.run(request=run_request)))
        
        return callback
    