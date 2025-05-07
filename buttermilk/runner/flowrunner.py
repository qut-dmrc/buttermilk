import asyncio
import importlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from buttermilk._core.config import SaveInfo
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.types import RunRequest
from buttermilk.api.flow import JobQueueClient
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
        initialized_flows: dict[str, OrchestratorProtocol] = {}

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

    async def pull_and_run_task(self) -> None:
        """Pull tasks from the queue and run them."""
        if self.queue_manager is None:
            self.queue_manager = JobQueueClient()

            # Pull task from the queue
            request = await self.queue_manager.pull_single_task()

            if request:
                # Run the task
                await self.run_flow(request, wait_for_completion=True)

    async def run_flow(self,
                      run_request: RunRequest,
                      wait_for_completion: bool = False,
                      **kwargs) -> Any:
        """Run a flow based on its configuration and a request.
        
        Args:
            run_request: The request containing input parameters
            wait_for_completion: If True, await the flow's completion before returning.
                                 If False (default), start the flow as a background
                                 task and return immediately.
            history: Optional conversation history (for chat-based interfaces)
            **kwargs: Additional keyword arguments for orchestrator instantiation
            
        Returns:
            If wait_for_completion is True, returns the result of the orchestrator run.
            If wait_for_completion is False, returns a callback function.
        
        Raises:
            ValueError: If orchestrator isn't specified or unknown

        """
        orchestrator = self.flows[run_request.flow]

        # Type safety: The orchestrator will be an Orchestrator instance at runtime,
        # even though the flows dict is typed with the more general OrchestratorProtocol
        callback = orchestrator._make_publish_callback()  # type: ignore

        # ======== MAJOR EVENT: FLOW STARTING ========
        # Log detailed information about flow start
        logger.info(f"🚀 FLOW STARTING: '{run_request.flow}' (ID: {run_request.job_id}).\n📋 RunRequest: {run_request.model_dump_json(indent=2)}\n⚙️ Source: {', '.join(run_request.source) if run_request.source else 'direct'}\n✅ New flow instance created - all state has been reset")

        # Create the task
        task = asyncio.create_task(orchestrator.run(request=run_request))  # type: ignore

        if wait_for_completion:
            # Wait for the task and return its result
            result = await task
            return result
        # Add task to list and return callback for non-blocking execution
        self.tasks.append(task)
        return callback
