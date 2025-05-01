import asyncio
import importlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from buttermilk._core.config import SaveInfo
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.types import RunRequest
from buttermilk.bm import BM, logger


class FlowRunner(BaseModel):
    """Centralized service for running flows across different entry points.
    
    Handles orchestrator instantiation and execution in a consistent way, regardless
    of whether the flow is started from CLI, API, Slackbot, or Pub/Sub.
    """

    bm: BM
    flows: dict[str, OrchestratorProtocol]  # Dictionary of instantiated flow orchestrators.
    flow_configs: dict[str, OrchestratorProtocol] = Field(default_factory=dict)  # Original flow configurations

    save: SaveInfo
    tasks: list = Field(default=[])
    ui: str
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def instantiate_orchestrators(self) -> "FlowRunner":
        """Initialize orchestrator instances from their configuration."""
        initialized_flows = {}

        for flow_name, flow_config in self.flows.items():
            if isinstance(flow_config, Orchestrator):
                # Already an instantiated orchestrator
                initialized_flows[flow_name] = flow_config
            else:
                # Extract orchestrator class path
                orchestrator_path = flow_config.orchestrator
                module_name, class_name = orchestrator_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                orchestrator_cls = getattr(module, class_name)

                # Create orchestrator instance with config
                config = flow_config if isinstance(flow_config, dict) else flow_config.model_dump()
                initialized_flows[flow_name] = orchestrator_cls(**config)

        self.flow_configs = self.flows
        self.flows = initialized_flows
        return self

    async def run_flow(self,
                        run_request: RunRequest,
                      **kwargs) -> Any:
        """Run a flow based on its configuration and a request.
        
        Args:
            run_request: The request containing input parameters
            history: Optional conversation history (for chat-based interfaces)
            **kwargs: Additional keyword arguments for orchestrator instantiation
            
        Returns:
            Task object if running asynchronously in background, otherwise None
        
        Raises:
            ValueError: If orchestrator isn't specified or unknown

        """
        orchestrator = self.flows[run_request.flow].orchestrator

        callback = orchestrator._make_publish_callback()

        logger.info(f"Starting flow '{run_request.flow}' with orchestrator '{orchestrator_cls.__name__}'")
        self.tasks.append(asyncio.create_task(orchestrator.run(request=run_request)))

        return callback
