import asyncio
from collections.abc import Mapping
from typing import Literal, Optional, Dict, Any, Union, Type

from pydantic import BaseModel, ConfigDict

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
    flows: Mapping[str, OrchestratorProtocol]  # Dictionary of configured flow orchestrators.
    ui: Literal["console", "api", "pub/sub", "slackbot"]  # The selected UI mode.
    save: SaveInfo
    llms: Mapping[str, str|list[str]]

    # Optional command-line overrides for the 'console' UI mode.
    record_id: str = ""
    flow: str = ""
    uri: str = ""
    prompt: str = ""

    model_config = ConfigDict(extra="allow")
    
    async def run_flow(self, 
                        flow_name: str,
                        run_request: Optional[RunRequest] = None,
                      history: Optional[list] = None,
                      **kwargs) -> Union[asyncio.Task, None]:
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
            
        orchestrator_cls = globals()[self.flows[flow_name]._target_]
        config = self.flows[flow_name].model_dump()
        # Create orchestrator instance
        orchestrator = orchestrator_cls(bm=self.bm, **config)
        
        logger.info(f"Starting flow '{flow_name}' with orchestrator '{orchestrator_cls.__name__}'")
    
        # Run synchronously and wait for completion
        await orchestrator.run(request=run_request)
        return None