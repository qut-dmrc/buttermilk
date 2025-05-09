import asyncio
import importlib
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from buttermilk._core.config import SaveInfo
from buttermilk import logger
from buttermilk._core.orchestrator import OrchestratorProtocol
from buttermilk._core.types import RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.bm import BM, logger


class FlowRunContext(BaseModel):
    """Encapsulates all state for a single flow run."""

    flow_name: str
    orchestrator: OrchestratorProtocol
    task: Any = None
    result: Any = None
    status: str = "pending"


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
    ui: str  # The UI mode (console, api, slackbot, etc.)
    ui_type: Optional[str] = Field(default=None)  # The UI implementation type (web, console, slack, etc.)
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def instantiate_orchestrators(self) -> "FlowRunner":
        """Initialize orchestrator instances from their configuration."""
        initialized_flows: dict[str, OrchestratorProtocol] = {}

        for flow_name, flow_config in self.flows.items():
            # Extract orchestrator class path
            orchestrator_path = flow_config.orchestrator
            module_name, class_name = orchestrator_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            orchestrator_cls = getattr(module, class_name)

            # Create orchestrator instance with config
            config = flow_config if isinstance(flow_config, dict) else flow_config.model_dump()
            
            # If we have a ui_type parameter and there's an interface agent in the observers,
            # make sure it gets the ui_type value
            if self.ui_type and "observers" in config:
                # Check if we have interface agents in the observers
                interface_agents = []
                for agent_group in config.get("observers", {}).keys():
                    if agent_group.startswith("interface/"):
                        interface_agents.append(agent_group)
                
                # If we found interface agents, update them with the ui_type
                if interface_agents:
                    logger.debug(f"Setting ui_type={self.ui_type} for interface agents: {interface_agents}")
                    for agent_key in interface_agents:
                        # Add ui_type to the agent parameters if it's a UIProxyAgent
                        agent_config = config["observers"][agent_key]
                        for role_name, role_config in agent_config.items():
                            if role_config.get("agent_obj") == "UIProxyAgent":
                                role_config["ui_type"] = self.ui_type
                                logger.debug(f"Set ui_type={self.ui_type} for {agent_key}.{role_name}")
            
            initialized_flows[flow_name] = orchestrator_cls(**config)

        self.flow_configs = self.flows
        self.flows = initialized_flows
        return self

    async def pull_and_run_task(self) -> None:
        """Pull tasks from the queue and run them."""
        # Initialize the queue_manager if needed
        queue_manager = getattr(self, "queue_manager", None)
        if queue_manager is None:
            self.queue_manager = JobQueueClient()
        
        # Pull task from the queue
        request = await self.queue_manager.pull_single_task()

        if request:
            # Run the task with a fresh orchestrator
            logger.info(f"Running task from queue: {request.flow} (Job ID: {request.job_id})")
            await self.run_flow(request, wait_for_completion=True)
        else:
            logger.debug("No tasks available in the queue")

    def _create_fresh_orchestrator(self, flow_name: str) -> OrchestratorProtocol:
        """Create a completely fresh orchestrator instance.
        
        Args:
            flow_name: The name of the flow to create an orchestrator for
            
        Returns:
            A new orchestrator instance with fresh state
            
        Raises:
            ValueError: If flow_name doesn't exist in flow_configs

        """
        if flow_name not in self.flow_configs:
            raise ValueError(f"Flow '{flow_name}' not found. Available flows: {list(self.flow_configs.keys())}")

        flow_config = self.flow_configs[flow_name]

        # Extract orchestrator class path
        orchestrator_path = flow_config.orchestrator
        module_name, class_name = orchestrator_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        orchestrator_cls = getattr(module, class_name)

        # Create a fresh config copy to avoid shared state
        config = flow_config.model_dump() if hasattr(flow_config, "model_dump") else dict(flow_config)

        # Create and return a fresh instance
        return orchestrator_cls(**config)

    async def _cleanup_flow_context(self, context: FlowRunContext) -> None:
        """Clean up resources associated with a flow run.
        
        Args:
            context: The flow run context to clean up

        """
        # Implement cleanup logic for the orchestrator if available
        cleanup_method = getattr(context.orchestrator, "cleanup", None)
        if cleanup_method is not None and callable(cleanup_method):
            try:
                result = cleanup_method()
                # Handle case where cleanup might be async or not
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Error during orchestrator cleanup: {e}")

        # Set status to completed
        context.status = "completed"

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
        # Create a fresh orchestrator instance
        fresh_orchestrator = self._create_fresh_orchestrator(run_request.flow)

        # Create a context for this specific run
        context = FlowRunContext(
            flow_name=run_request.flow,
            orchestrator=fresh_orchestrator,
        )

        # Type safety: The orchestrator will be an Orchestrator instance at runtime,
        # even though the flows dict is typed with the more general OrchestratorProtocol
        callback = fresh_orchestrator._make_publish_callback()  # type: ignore

        # ======== MAJOR EVENT: FLOW STARTING ========
        # Log detailed information about flow start
        logger.info(f"🚀 FLOW STARTING: '{run_request.flow}' (ID: {run_request.job_id}).\n📋 RunRequest: {run_request.model_dump_json(indent=2)}\n⚙️ Source: {', '.join(run_request.source) if run_request.source else 'direct'}\n✅ New flow instance created - all state has been reset")

        # Create the task
        context.task = asyncio.create_task(fresh_orchestrator.run(request=run_request))  # type: ignore

        try:
            if wait_for_completion:
                # Wait for the task and return its result
                context.result = await context.task
                context.status = "completed"
                return context.result
            # Add task to list and return callback for non-blocking execution
            # Note: We still keep this for backward compatibility, but each task is now tied to a fresh instance
            self.tasks.append(context.task)
            return callback
        except Exception as e:
            context.status = "failed"
            logger.error(f"Error running flow '{run_request.flow}': {e}")
            raise
        finally:
            if wait_for_completion:
                # Clean up after completion if we were waiting
                await self._cleanup_flow_context(context)
