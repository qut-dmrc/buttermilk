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
from autogen_core import (
    AgentId,
    AgentType,
    ClosureAgent,
    ClosureContext,  # Represents a registered type of agent in the runtime.
    DefaultInterventionHandler,
    DefaultTopicId,
    DropMessage,  # A standard implementation for topic identifiers.
    MessageContext,
    SingleThreadedAgentRuntime,  # The core runtime managing agents and messages.
    TopicId,  # Abstract base class for topic identifiers.
    TypeSubscription,  # Defines a subscription based on message type and agent type.
)
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk import (
    buttermilk as bm,  # Global Buttermilk instance
    logger,
)
from buttermilk._core import (
    AllMessages,
    StepRequest,
)
from buttermilk._core.agent import Agent, ProcessingError
from buttermilk._core.constants import MANAGER
from buttermilk._core.contract import (
    ConductorRequest,
    FlowEvent,
    FlowMessage,
    ManagerMessage,
    TaskProcessingComplete,
)
from buttermilk._core.exceptions import FatalError
from buttermilk._core.orchestrator import Orchestrator  # Base class for orchestrators.
from buttermilk._core.types import RunRequest

# AutogenAgentAdapter no longer needed - Agent now inherits from RoutedAgent directly


class InterruptHandler(BaseModel):
    """A simple handler for managing interrupts in the flow."""

    interrupt: asyncio.Event = Field(default_factory=asyncio.Event)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any | type[DropMessage]:
        """Called when a message is published to the AgentRuntime using :meth:`autogen_core.base.AgentRuntime.publish_message`."""
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

    async def on_send(self, message: Any, *, message_context: MessageContext, recipient: AgentId) -> Any | type[DropMessage]:
        """Called when a message is submitted to the AgentRuntime using :meth:`autogen_core.base.AgentRuntime.send_message`."""
        return message

    async def on_response(self, message: Any, *, sender: AgentId, recipient: AgentId | None) -> Any | type[DropMessage]:
        """Called when a response is received by the AgentRuntime from an Agent's message handler returning a value."""
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
    _pending_messages: list[tuple[FlowMessage, TopicId]] = PrivateAttr(default_factory=list)
    _is_initialized: bool = PrivateAttr(default=False)

    # Dynamically generates a unique topic ID for this specific orchestrator run.
    # Ensures messages within this run don't interfere with other concurrent runs.
    _topic: TopicId = PrivateAttr(default=None)

    async def _setup(self, request: RunRequest) -> tuple[TerminationHandler, InterruptHandler]:
        """Initializes the Autogen runtime and registers all configured agents."""
        # Initialize the topic ID if not already set
        if self._topic is None:
            self._topic = DefaultTopicId(type=f"{bm.name}-{bm.job}-{shortuuid.uuid()[:8]}")

        msg = f"Setting up AutogenOrchestrator for topic: {self._topic.type}"
        logger.info(f"[AutogenOrchestrator._setup] {msg} (callback_to_ui: {'set' if request.callback_to_ui else 'not set'})")

        termination_handler = TerminationHandler()
        interrupt_handler = InterruptHandler()
        self._runtime = SingleThreadedAgentRuntime(intervention_handlers=[termination_handler, interrupt_handler])

        # Start the Autogen runtime's processing loop in the background.
        self._runtime.start()

        # Register Buttermilk agents (wrapped in Adapters) with the Autogen runtime.
        await self._register_agents(params=request)

        await self.register_ui(callback_to_ui=request.callback_to_ui)

        # Give the UI registration time to complete
        await asyncio.sleep(2)

        # Send a broadcast message to initialize all agents subscribed to the group chat
        logger.info(f"Broadcasting initialization message to topic '{self._topic.type}' to wake up all agents")
        await self._runtime.publish_message(
            FlowEvent(source="orchestrator", content="Initializing group chat participants"),
            topic_id=self._topic,
        )

        # Give agents a moment to initialize
        await asyncio.sleep(0.5)

        # Send a welcome message to the UI
        flow_event = FlowEvent(source="orchestrator", content=msg)
        topic = DefaultTopicId(type=MANAGER)
        logger.debug("[AutogenOrchestrator._setup] Publishing welcome message to MANAGER topic")
        await self._runtime.publish_message(flow_event, topic_id=topic)

        # Give the MANAGER a moment to process the message
        await asyncio.sleep(0.5)

        # Mark as initialized and process any pending messages
        self._is_initialized = True

        # Process any messages that were queued before initialization
        if self._pending_messages:
            logger.debug(f"[AutogenOrchestrator._setup] Processing {len(self._pending_messages)} pending messages")
            for pending_message, topic_id in self._pending_messages:
                await self._runtime.publish_message(pending_message, topic_id=topic_id)

        # Clear the pending messages
        self._pending_messages.clear()

        # Start up the host agent with participants and their tools
        logger.highlight(f"Sending ConductorRequest to topic '{self._topic}' with {len(self._participants)} participants: {list(self._participants.keys())}")
        conductor_request = ConductorRequest(
            inputs=request.model_dump(),
            participants=self._participants,
        )
        logger.debug(f"ConductorRequest details - participants: {conductor_request.participants}")
        await self._runtime.publish_message(
            conductor_request,
            topic_id=self._topic,
        )

        return termination_handler, interrupt_handler

    async def _register_agents(self, params: RunRequest) -> None:
        """Registers Buttermilk agents with the Autogen runtime.

        Iterates through the `self.agents` configuration, creating Agent
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
            **{v.role: v.description for k, v in self.observers.items()},
        }
        logger.info(f"Created participants dictionary with {len(self._participants)} entries: {list(self._participants.keys())}")

        for role_name, step_config in itertools.chain(self.agents.items(), self.observers.items()):
            registered_for_role = []
            # `get_configs` yields tuples of (AgentClass, agent_variant_config)
            for agent_cls, variant_config in step_config.get_configs(params=params, flow_default_params=self.parameters):
                # Define a factory function required by Autogen's registration.
                if isinstance(agent_cls, type(Agent)):
                    config_with_session = {**variant_config.model_dump(), "session_id": params.session_id}

                    # Create factory function for the agent
                    def agent_factory(
                        orchestrator_ref,  # Reference to orchestrator for registration
                        cfg: dict = config_with_session,
                        cls: type[Agent] = agent_cls,
                    ) -> Agent:
                        # Create the agent instance
                        agent_instance = cls(**cfg)
                        # Agent registration is now handled by the Agent class itself
                        return agent_instance

                    # Register the agent factory with the runtime.
                    # The runtime will call this factory to create agent instances.
                    agent_type: AgentType = await Agent.register(
                        runtime=self._runtime,
                        type=variant_config.agent_id,  # Use the specific variant ID for registration
                        factory=lambda orch=self, v_cfg=config_with_session, a_cls=agent_cls: agent_factory(
                            orch, cfg=v_cfg, cls=a_cls,
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
                # Use the actual agent role from config, not the dictionary key
                actual_role = step_config.role.upper()
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=actual_role,
                        agent_type=agent_type,
                    ),
                )
                logger.debug(f"Registered agent: ID='{variant_config.agent_name}', Role='{actual_role}', Type='{agent_type}'. Subscribed to topics: '{self._topic.type}', '{actual_role}'")

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
        if not callback_to_ui:
            logger.warning("No UI callback provided. Messages will not be sent to the UI.")

        async def output_result(_ctx: ClosureContext, message: AllMessages, ctx: MessageContext) -> None:
            try:
                result = await callback_to_ui(message)
            except Exception as e:
                logger.error(f"[MANAGER ClosureAgent] ❌ Error calling callback_to_ui: {e}", exc_info=True)

        # Register the closure function as an agent named MANAGER.
        logger.debug(f"[AutogenOrchestrator.register_ui] Attempting to register ClosureAgent with type: {MANAGER}")
        await ClosureAgent.register_closure(
            runtime=self._runtime,
            type=MANAGER,  # Agent ID/Name.
            closure=output_result,  # The async function to handle messages.
            subscriptions=lambda: [
                TypeSubscription(
                    topic_type=MANAGER,  # Subscribe to the MANAGER topic
                    agent_type=MANAGER,
                ),
                TypeSubscription(
                    topic_type=self._topic.type,  # Main group chat topic
                    agent_type=MANAGER,
                ),
            ],
            # If a message arrives that isn't handled, just ignore it silently.
            # unknown_type_policy="ignore",
        )
        logger.debug(f"[AutogenOrchestrator.register_ui] ClosureAgent registered successfully for type: {MANAGER}")

        # Give the agent time to fully register
        await asyncio.sleep(0.5)
        logger.debug("[AutogenOrchestrator.register_ui] Registration complete after delay")

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
                logger.debug(f"[AutogenOrchestrator._run] Calling _setup with request.callback_to_ui: {request.callback_to_ui is not None}")
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
                        logger.debug(f"[AutogenOrchestrator._run] Publishing record: {record}")
                        await self._runtime.publish_message(record, topic_id=self._topic)

            # 3. Wait for termination.
            while True:
                try:
                    if termination_handler.has_terminated:
                        logger.info("Termination message received.")
                        # Send flow_completed event before TaskProcessingComplete
                        await self._runtime.publish_message(
                            FlowEvent(source="orchestrator", content="flow_completed"),
                            topic_id=DefaultTopicId(type=MANAGER),
                        )
                        # Publish a TaskProcessingComplete message to the UI
                        logger.debug("[AutogenOrchestrator._run] Publishing TaskProcessingComplete message.")
                        logger.debug("[AutogenOrchestrator._run] Publishing TaskProcessingComplete message to MANAGER topic.")
                        await self._runtime.publish_message(
                            TaskProcessingComplete(
                                agent_id="orchestrator",
                                role="orchestrator",
                                status="COMPLETED",
                                message="Flow completed successfully.",
                                more_tasks_remain=False,
                            ),
                            topic_id=DefaultTopicId(type=MANAGER),
                        )
                        logger.debug("[AutogenOrchestrator._run] TaskProcessingComplete message published.")
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
            # Cleanup is now handled by the orchestrator lifecycle management
            pass

    def make_publish_callback(self) -> Callable[[FlowMessage], Awaitable[None]]:
        """Creates an asynchronous callback function for the UI to use.

        Returns:
            An async callback function that takes a `FlowMessage` and publishes it.

        """
        async def publish_callback(message: FlowMessage) -> None:
            logger.debug(f"[AutogenOrchestrator.make_publish_callback] Publishing message to runtime: {message}")

            # If runtime is not initialized yet, queue the message
            if not self._is_initialized or not hasattr(self, "_runtime") or self._runtime is None:
                logger.info(f"[AutogenOrchestrator] Runtime not initialized yet, queueing message: {message}")
                self._pending_messages.append((message, self._topic))
                return

            await self._runtime.publish_message(
                message,
                topic_id=self._topic,
            )

        return publish_callback
