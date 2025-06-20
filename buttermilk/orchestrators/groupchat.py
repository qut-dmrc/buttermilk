"""Implements an Orchestrator using the autogen-core library for managing multi-agent interactions.

This module defines `AutogenOrchestrator`, which leverages Autogen's agent registration,
message routing (via topics and subscriptions), and runtime management to execute
complex workflows involving multiple LLM agents. It's designed to be configured via Hydra
and integrates with the Buttermilk agent and contract system.
"""

import asyncio
import itertools
from collections.abc import Awaitable, Callable  # Added type hints for clarity
from typing import Any

import shortuuid
import weave
from autogen_core import (
    AgentType,
    ClosureAgent,
    ClosureContext,  # Represents a registered type of agent in the runtime.
    DefaultInterventionHandler,
    DefaultTopicId,  # A standard implementation for topic identifiers.
    MessageContext,
    SingleThreadedAgentRuntime,  # The core runtime managing agents and messages.
    TopicId,  # Abstract base class for topic identifiers.
    TypeSubscription,  # Defines a subscription based on message type and agent type.
)
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk import buttermilk as bm  # Global Buttermilk instance
from buttermilk._core import (  # noqa
    BM,
    AllMessages,
    StepRequest,
    # logger, # logger is imported below
)
from buttermilk._core.agent import Agent, ProcessingError
from buttermilk._core.config import AgentConfig
from buttermilk._core.constants import CONDUCTOR, MANAGER
from buttermilk._core.contract import (
    ConductorRequest,
    FlowEvent,
    FlowMessage,
    ManagerMessage,
)
from buttermilk._core.exceptions import FatalError
from buttermilk._core.log import logger  # noqa
from buttermilk._core.orchestrator import Orchestrator  # Base class for orchestrators.
from buttermilk._core.types import RunRequest
from buttermilk.libs.autogen import AutogenAgentAdapter


class InterruptHandler(BaseModel):
    """A simple handler for managing interrupts in the flow."""

    interrupt: asyncio.Event = Field(default_factory=asyncio.Event)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, ManagerMessage):
            if message.interrupt:
                # Pause the flow
                logger.info(f"Manager interrupt message received: {message}")
                self.interrupt.set()
            elif self.interrupt.is_set():
                # Resume the flow
                logger.info(f"Manager resume message received: {message}")
                self.interrupt.clear()
        return message


class TerminationHandler(DefaultInterventionHandler):
    def __init__(self) -> None:
        self._termination_value: StepRequest | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, StepRequest):
            if message.role == "END":
                # This is a termination message
                logger.info(f"Termination message received: {message}")
                self._termination_value = message
        return message

    def request_termination(self):
        """Signal that the flow should terminate"""
        self._termination_value = StepRequest(role="END", content="Termination requested")

    @property
    def termination_value(self) -> StepRequest | None:
        return self._termination_value

    @property
    def has_terminated(self) -> bool:
        return self._termination_value is not None


class AutogenOrchestrator(Orchestrator):
    """Orchestrates multi-agent workflows using the autogen-core library.

    This orchestrator manages a `SingleThreadedAgentRuntime` to host and coordinate
    Buttermilk agents adapted for Autogen. It uses Autogen's topic-based pub/sub
    system for message routing between agents. A central `CONDUCTOR` agent is
    typically used to determine the sequence of steps in the workflow. It also
    supports interaction with a `MANAGER` agent (often representing a human user or UI)
    for approvals or feedback.

    Inherits configuration fields (agents, parameters, data, etc.) from the base `Orchestrator`.

    Attributes:
        _runtime: The underlying Autogen runtime instance.
        _agent_types: Maps role names (UPPERCASE) to lists of registered Autogen AgentTypes and variant configs.
        _topic: The main topic ID for this specific group chat instance, generated uniquely per run.

    """

    # Private attributes managed internally. Use PrivateAttr for Pydantic integration.
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
    _agent_types: dict[str, list[tuple[AgentType, Any]]] = PrivateAttr(default_factory=dict)
    _participants: dict[str, str] = PrivateAttr()
    _agent_registry: dict[str, Agent] = PrivateAttr(default_factory=dict)

    # Dynamically generates a unique topic ID for this specific orchestrator run.
    # Ensures messages within this run don't interfere with other concurrent runs.
    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(type=f"{bm.name}-{bm.job}-{shortuuid.uuid()[:8]}"),
    )

    def _register_buttermilk_agent_instance(self, agent_id: str, agent_instance: Agent) -> None:
        if agent_id in self._agent_registry:
            logger.warning(f"Agent with ID '{agent_id}' already exists in the registry. Overwriting.")
        self._agent_registry[agent_id] = agent_instance
        logger.debug(f"Registered Buttermilk agent instance '{agent_instance.agent_name}' with ID '{agent_id}' to orchestrator registry.")

    async def _setup(self, request: RunRequest) -> tuple[TerminationHandler, InterruptHandler]:
        """Initializes the Autogen runtime and registers all configured agents."""
        msg = f"Setting up AutogenOrchestrator for topic: {self._topic.type}"
        logger.info(msg)

        termination_handler = TerminationHandler()
        interrupt_handler = InterruptHandler()
        self._runtime = SingleThreadedAgentRuntime(intervention_handlers=[termination_handler, interrupt_handler])

        # Start the Autogen runtime's processing loop in the background.
        self._runtime.start()
        logger.debug("Autogen runtime started.")

        # Register Buttermilk agents (wrapped in Adapters) with the Autogen runtime.
        await self._register_agents(params=request)

        await self.register_ui(callback_to_ui=request.callback_to_ui)

        # does it need a second to spin up?
        await asyncio.sleep(1)

        # Send a broadcast message to initialize all agents subscribed to the group chat
        logger.info(f"Broadcasting initialization message to topic '{self._topic.type}' to wake up all agents")
        await self._runtime.publish_message(
            FlowEvent(source="orchestrator", content="Initializing group chat participants"),
            topic_id=self._topic
        )

        # Give agents a moment to initialize
        await asyncio.sleep(0.5)

        # Send a welcome message to the UI
        await self._runtime.publish_message(FlowEvent(source="orchestrator", content=msg), topic_id=DefaultTopicId(type=MANAGER))

        # Collect tool definitions from registered agents
        participant_tools = {}
        logger.info(f"Collecting tools from {len(self._agent_registry)} registered agents")
        for agent_id, agent_instance in self._agent_registry.items():
            if hasattr(agent_instance, 'role') and hasattr(agent_instance, 'get_tool_definitions'):
                role = agent_instance.role.upper()
                try:
                    tool_defs = agent_instance.get_tool_definitions()
                    if tool_defs:
                        # Convert tool definitions to dict format
                        participant_tools[role] = [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.input_schema,
                                "output_schema": tool.output_schema
                            }
                            for tool in tool_defs
                        ]
                        logger.info(f"Collected {len(tool_defs)} tool definitions from {role}: {[t['name'] for t in participant_tools[role]]}")
                    else:
                        logger.debug(f"No tool definitions found for {role}")
                except Exception as e:
                    logger.warning(f"Failed to get tool definitions from {role}: {e}")
            else:
                logger.debug(f"Agent {agent_id} does not have role or get_tool_definitions")

        # Start up the host agent with participants and their tools
        logger.info(f"Sending ConductorRequest to topic '{CONDUCTOR}' with {len(self._participants)} participants: {list(self._participants.keys())} and tools: {list(participant_tools.keys())}")
        conductor_request = ConductorRequest(
            inputs=request.model_dump(), 
            participants=self._participants,
            participant_tools=participant_tools
        )
        logger.debug(f"ConductorRequest details - participants: {conductor_request.participants}")
        logger.debug(f"ConductorRequest details - tools: {participant_tools}")
        await self._runtime.publish_message(
            conductor_request, 
            topic_id=DefaultTopicId(type=CONDUCTOR)
        )

        return termination_handler, interrupt_handler

    async def _register_agents(self, params: RunRequest) -> None:
        """Registers Buttermilk agents (via Adapters) with the Autogen runtime.

        Iterates through the `self.agents` configuration, creating AutogenAgentAdapter
        instances for each agent variant and registering them with the runtime.
        Sets up subscriptions so agents listen on the main group chat topic and
        potentially role-specific topics.
        """
        logger.debug("Registering agents with Autogen runtime...")

        # Add flow's static parameters to the request parameters
        # Create list of participants in the group chat - include both agents AND observers
        logger.info(f"Creating participants from {len(self.agents)} agents and {len(self.observers)} observers")
        self._participants = {
            **{v.role: v.description for k, v in self.agents.items()},
            **{v.role: v.description for k, v in self.observers.items()}
        }
        logger.info(f"Created participants dictionary with {len(self._participants)} entries: {list(self._participants.keys())}")

        for role_name, step_config in itertools.chain(self.agents.items(), self.observers.items()):
            registered_for_role = []
            # `get_configs` yields tuples of (AgentClass, agent_variant_config)
            for agent_cls, variant_config in step_config.get_configs(params=params, flow_default_params=self.parameters):
                # Define a factory function required by Autogen's registration.
                if isinstance(agent_cls, type(Agent)):
                    config_with_session = {**variant_config.model_dump(), "session_id": params.session_id}

                    # This function creates an instance of the AutogenAgentAdapter,
                    # wrapping the actual Buttermilk agent logic.
                    # It captures loop variables (variant_config, agent_cls, self._topic.type)
                    # to ensure the correct configuration is used when the factory is called.

                    def agent_factory(
                        orchestrator_ref,  # New parameter for self
                        cfg: dict = config_with_session,
                        cls: type[Agent] = agent_cls,
                        topic_type: str = self._topic.type,
                    ):
                        # The adapter will need to accept 'registration_callback'
                        return AutogenAgentAdapter(
                            agent_cfg=cfg,
                            agent_cls=cls,
                            topic_type=topic_type,
                            registration_callback=orchestrator_ref._register_buttermilk_agent_instance,  # Pass the method
                        )

                    # Register the adapter factory with the runtime.
                    # `variant_config.id` should be a unique identifier for this specific agent instance/variant.
                    agent_type: AgentType = await AutogenAgentAdapter.register(
                        runtime=self._runtime,
                        type=variant_config.agent_id,  # Use the specific variant ID for registration
                        factory=lambda orch=self, v_cfg=config_with_session, a_cls=agent_cls, t_type=self._topic.type: agent_factory(
                            orch, cfg=v_cfg, cls=a_cls, topic_type=t_type,
                        ),
                    )
                else:
                    # Register the adapter factory with the runtime.
                    agent_type: AgentType = await agent_cls.register(
                        runtime=self._runtime,
                        type=variant_config.agent_id,  # Use the specific variant ID for registration
                        factory=lambda params=variant_config.parameters, cls=agent_cls:  cls(**params),
                    )

                # Subscribe the newly registered agent type to the main group chat topic.
                # This allows it to receive general messages sent to the group.
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=self._topic.type,  # Main group chat topic
                        agent_type=agent_type,
                    ),
                )

                # Also subscribe the agent to a topic specific to its role (e.g., "JUDGE", "SCORER").
                # This allows targeted messages to be sent directly to agents fulfilling that specific role.
                role_topic_type = role_name.upper()
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=role_topic_type,
                        agent_type=agent_type,
                    ),
                )
                logger.debug(f"Registered agent adapter: ID='{variant_config.agent_name}', Role='{role_name}', Type='{agent_type}'. Subscribed to topics: '{self._topic.type}', '{role_topic_type}'")

                registered_for_role.append((agent_type, variant_config))

            # Store the list of (AgentType, variant_config) tuples for this role.
            # Use uppercase role name as the key, consistent with topic subscription.
            self._agent_types[role_name.upper()] = registered_for_role
            logger.debug(f"Registered {len(registered_for_role)} agent variants for role '{role_name}'.")

    async def register_ui(self, callback_to_ui: Callable[..., Awaitable[None]]) -> None:
        """Registers a callback function for the groupchat to send messages to the UI.

        Args:
            callback_to_ui: A callable that takes a message and sends it to the UI.

        """
        if callback_to_ui:
            logger.debug("Registering UI callback...")
        else:
            logger.warning("No UI callback provided. Messages will not be sent to the UI.")

        async def output_result(_ctx: ClosureContext, message: AllMessages, ctx: MessageContext) -> None:
            if callback_to_ui is not None:
                logger.debug(f"Sending message to UI: {message}")
                await callback_to_ui(message)
            else:
                logger.debug(f"[{self.trace_id}] {message}")

        # Register the closure function as an agent named MANAGER.
        await ClosureAgent.register_closure(
            runtime=self._runtime,
            type=MANAGER,  # Agent ID/Name.
            closure=output_result,  # The async function to handle messages.
            subscriptions=lambda: [
                TypeSubscription(
                    topic_type=self._topic.type,  # Subscribe to the main group chat topic
                    agent_type=MANAGER,
                ),
            ],
            # If a message arrives that isn't handled, just ignore it silently.
            # unknown_type_policy="ignore",
        )

    async def _cleanup(self) -> None:
        """Cleans up resources with timeout and verification."""
        logger.debug("Cleaning up AutogenOrchestrator...")
        cleanup_timeout = 15.0  # 15 second timeout for cleanup

        try:
            # Cancel all registered agent tasks first
            logger.debug("Cleaning up registered agents...")
            for agent_id, agent_instance in self._agent_registry.items():
                try:
                    if hasattr(agent_instance, "cleanup"):
                        cleanup_result = agent_instance.cleanup()
                        if asyncio.iscoroutine(cleanup_result):
                            await asyncio.wait_for(cleanup_result, timeout=5.0)
                    logger.debug(f"Cleaned up agent: {agent_id}")
                except TimeoutError:
                    logger.warning(f"Timeout cleaning up agent {agent_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up agent {agent_id}: {e}")

            # Also cleanup any agent adapters in the runtime
            if hasattr(self, "_runtime") and hasattr(self._runtime, "_agents"):
                logger.debug("Cleaning up agent adapters...")
                for agent_type, agents in self._runtime._agents.items():
                    for agent in agents:
                        if hasattr(agent, "cleanup"):
                            try:
                                await asyncio.wait_for(agent.cleanup(), timeout=5.0)
                                logger.debug(f"Cleaned up adapter for agent type: {agent_type}")
                            except TimeoutError:
                                logger.warning(f"Timeout cleaning up adapter for {agent_type}")
                            except Exception as e:
                                logger.warning(f"Error cleaning up adapter for {agent_type}: {e}")

            # Clear registries
            self._agent_registry.clear()
            self._agent_types.clear()

            # Stop the runtime with timeout
            if hasattr(self, "_runtime") and self._runtime._run_context:
                logger.debug("Stopping Autogen runtime...")
                try:
                    cleanup_task = asyncio.create_task(self._runtime.close())
                    await asyncio.wait_for(cleanup_task, timeout=cleanup_timeout)
                    logger.info("Autogen runtime stopped successfully")
                except TimeoutError:
                    logger.error(f"Autogen runtime cleanup timeout after {cleanup_timeout}s")
                    # Force cleanup
                    await self._force_cleanup()
                except Exception as e:
                    logger.error(f"Error during runtime cleanup: {e}")
                    await self._force_cleanup()

            # Verify cleanup
            await self._verify_cleanup()

            # Print weave link if available
            try:
                call = weave.get_current_call()
                if call and hasattr(call, "ui_url"):
                    logger.info(f"Tracing link: 🍩 {call.ui_url}")
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Fatal error during orchestrator cleanup: {e}")
            await self._force_cleanup()

    async def _force_cleanup(self) -> None:
        """Force cleanup when normal cleanup fails."""
        logger.warning("Performing force cleanup of AutogenOrchestrator")
        try:
            # Clear all internal state
            self._agent_registry.clear()
            self._agent_types.clear()

            # Try to forcefully stop runtime
            if hasattr(self, "_runtime"):
                try:
                    # Set runtime context to None to prevent further operations
                    if hasattr(self._runtime, "_run_context"):
                        self._runtime._run_context = None
                except Exception as e:
                    logger.warning(f"Error during force cleanup: {e}")

        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")

    async def _verify_cleanup(self) -> None:
        """Verify that cleanup was successful."""
        issues = []

        # Check that registries are cleared
        if self._agent_registry:
            issues.append(f"{len(self._agent_registry)} agents still in registry")
        if self._agent_types:
            issues.append(f"{len(self._agent_types)} agent types still registered")

        # Check runtime state
        if hasattr(self, "_runtime") and self._runtime._run_context:
            issues.append("Runtime context still active")

        if issues:
            logger.warning(f"Cleanup verification failed: {', '.join(issues)}")
        else:
            logger.debug("Cleanup verification passed")

    async def _run(self, request: RunRequest, flow_name: str = "") -> None:
        """Simplified main execution loop for the orchestrator.

        This version delegates most of the substantive flow control to the
        host (CONDUCTOR) agent. The orchestrator now acts mainly as a message
        bus between agents, handling only the technical aspects of execution:
        1. Setting up the runtime and agents
        2. Loading initial data if needed
        3. Getting step suggestions from the host agent
        4. Getting user confirmation if needed
        5. Executing the steps by sending messages

        All decisions about what steps to take, pacing, and agent coordination
        are delegated to the host agent.

        Args:
            request: An optional RunRequest containing initial data.

        """
        try:
            # 1. Setup the runtime and agents
            try:
                termination_handler, interrupt_handler = await self._setup(request)
            except Exception as e:
                logger.error(f"Error during setup: {e}")
                raise FatalError from e

            # 2. Load initial data if provided
            if request:
                await self._fetch_initial_records(request)  # Use helper for clarity
                if self._records:
                    for record in self._records:
                        # send each record to all clients
                        await self._runtime.publish_message(record, topic_id=self._topic)

            # 3. Wait for termination.
            while True:
                try:
                    if termination_handler.has_terminated:
                        logger.info("Termination message received.")
                        break
                    if interrupt_handler.interrupt.is_set():
                        logger.info("Flow is paused. Waiting for resume...")
                        while interrupt_handler.interrupt.is_set():
                            await asyncio.sleep(0.5)
                        logger.info("Flow resumed.")
                    await asyncio.sleep(0.1)

                except ProcessingError as e:
                    # Non-fatal error - let the host agent decide how to recover
                    logger.error(f"Error in execution: {e}")
                except (StopAsyncIteration, KeyboardInterrupt):
                    raise
                except FatalError:
                    raise
                except Exception as e:
                    raise FatalError from e

        except (KeyboardInterrupt):
            logger.info("Flow terminated by user.")
        except (FatalError, Exception) as e:
            logger.exception(f"Unexpected and unhandled fatal error: {e}", exc_info=True)
        finally:
            await self._cleanup()

    def make_publish_callback(self) -> Callable[[FlowMessage], Awaitable[None]]:
        """Creates an asynchronous callback function for the UI to use.

        Returns:
            An async callback function that takes a `FlowMessage` and publishes it.

        """
        async def publish_callback(message: FlowMessage) -> None:
            await self._runtime.publish_message(
                message,
                topic_id=self._topic,
            )

        return publish_callback

    def get_agent_config(self, agent_id: str) -> AgentConfig | None:
        """Retrieves the AgentConfig for a given agent_id from the registry.

        Args:
            agent_id: The ID of the agent to retrieve.

        Returns:
            The AgentConfig of the agent if found, otherwise None.

        """
        agent_instance = self._agent_registry.get(agent_id)
        if agent_instance:
            # The Agent class inherits from AgentConfig.
            # We can construct a clean AgentConfig from the instance.
            agent_config_fields = set(AgentConfig.model_fields.keys())
            config_data = {
                field: getattr(agent_instance, field)
                for field in agent_config_fields
                if hasattr(agent_instance, field)
            }
            return AgentConfig(**config_data)
        logger.info(f"Agent with ID '{agent_id}' not found in the orchestrator's agent registry.")
        return None
