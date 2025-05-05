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
    AgentType,  # Represents a registered type of agent in the runtime.
    DefaultTopicId,  # A standard implementation for topic identifiers.
    SingleThreadedAgentRuntime,  # The core runtime managing agents and messages.
    TopicId,  # Abstract base class for topic identifiers.
    TypeSubscription,  # Defines a subscription based on message type and agent type.
)
from pydantic import PrivateAttr

from buttermilk._core import AgentInput, AgentTrace
from buttermilk._core.agent import Agent
from buttermilk._core.constants import CONDUCTOR, MANAGER
from buttermilk._core.contract import (
    ConductorRequest,
    FlowMessage,
    ManagerMessage,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.orchestrator import Orchestrator  # Base class for orchestrators.
from buttermilk._core.types import RunRequest
from buttermilk.bm import bm, logger  # Core Buttermilk instance and logger.
from buttermilk.libs.autogen import AutogenAgentAdapter  # Adapter to wrap Buttermilk agents for Autogen.


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

    # Dynamically generates a unique topic ID for this specific orchestrator run.
    # Ensures messages within this run don't interfere with other concurrent runs.
    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(type=f"{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:8]}"),
    )

    async def _setup(self, request: RunRequest) -> None:
        """Initializes the Autogen runtime and registers all configured agents."""
        msg = f"Setting up AutogenOrchestrator for topic: {self._topic.type}"
        logger.info(msg)
        self._runtime = SingleThreadedAgentRuntime()

        # Start the Autogen runtime's processing loop in the background.
        self._runtime.start()

        # Register Buttermilk agents (wrapped in Adapters) with the Autogen runtime.
        await self._register_agents(params=request)

        logger.debug("Autogen runtime started.")

        # does it need a second to spin up?
        await asyncio.sleep(1)

        # Send a welcome message to the UI and start up the host agent
        await self._runtime.publish_message(ManagerMessage(content=msg), topic_id=DefaultTopicId(type=MANAGER))
        await self._runtime.publish_message(ConductorRequest(inputs=request.model_dump(), participants=self._participants), topic_id=DefaultTopicId(type=CONDUCTOR))

    async def _register_agents(self, params: RunRequest) -> None:
        """Registers Buttermilk agents (via Adapters) with the Autogen runtime.

        Iterates through the `self.agents` configuration, creating AutogenAgentAdapter
        instances for each agent variant and registering them with the runtime.
        Sets up subscriptions so agents listen on the main group chat topic and
        potentially role-specific topics.
        """
        logger.debug("Registering agents with Autogen runtime...")

        # Create list of participants in the group chat
        self._participants = {k: v.description for k, v in self.agents.items()}

        for role_name, step_config in itertools.chain(self.agents.items(), self.observers.items()):
            registered_for_role = []
            # `get_configs` yields tuples of (AgentClass, agent_variant_config)
            for agent_cls, variant_config in step_config.get_configs(params=params):
                # Define a factory function required by Autogen's registration.
                if isinstance(agent_cls, type(Agent)):
                    config_with_session = {**variant_config.model_dump(), "session_id": self.session_id}

                    # This function creates an instance of the AutogenAgentAdapter,
                    # wrapping the actual Buttermilk agent logic.
                    # It captures loop variables (variant_config, agent_cls, self._topic.type)
                    # to ensure the correct configuration is used when the factory is called.

                    def agent_factory(cfg=config_with_session, cls=agent_cls, topic_type=self._topic.type):
                        return AutogenAgentAdapter(
                            agent_cfg=cfg,
                            agent_cls=cls,
                            topic_type=topic_type,  # Pass the main topic type
                        )

                    # Register the adapter factory with the runtime.
                    # `variant_config.id` should be a unique identifier for this specific agent instance/variant.
                    agent_type: AgentType = await AutogenAgentAdapter.register(
                        runtime=self._runtime,
                        type=variant_config.agent_id,  # Use the specific variant ID for registration
                        factory=agent_factory,
                    )
                else:

                    # Register the adapter factory with the runtime.
                    agent_type: AgentType = await agent_cls.register(
                        runtime=self._runtime,
                        type=variant_config.agent_id,  # Use the specific variant ID for registration
                        factory=lambda params=variant_config.parameters, cls=agent_cls:  cls(**params),
                    )

                logger.debug(f"Registered agent adapter: ID='{variant_config.agent_id}', Role='{role_name}', Type='{agent_type}'")

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
                logger.debug(f"Agent type '{agent_type}' subscribed to topics: '{self._topic.type}', '{role_topic_type}'")

                registered_for_role.append((agent_type, variant_config))

            # Store the list of (AgentType, variant_config) tuples for this role.
            # Use uppercase role name as the key, consistent with topic subscription.
            self._agent_types[role_name.upper()] = registered_for_role
            logger.debug(f"Registered {len(registered_for_role)} agent variants for role '{role_name}'.")

    async def _cleanup(self) -> None:
        """Cleans up resources, primarily by stopping the Autogen runtime."""
        logger.debug("Cleaning up AutogenOrchestrator...")
        try:
            # Stop the runtime
            if self._runtime._run_context:
                # runtime is started
                await self._runtime.stop_when_idle()

            # Print weave link again
            call = weave.get_current_call()
            logger.info(f"Tracing link for weave: 🍩 {call.ui_url}")

        except Exception as e:
            logger.warning(f"Error during runtime cleanup: {e}")
        finally:
            await asyncio.sleep(2)  # Give it some time to properly shut down

    @weave.op
    async def _run(self, request: RunRequest | None = None) -> None:
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
            await self._setup(request or RunRequest())

            # 2. Load initial data if provided
            if request:
                await self._fetch_initial_records(request)  # Use helper for clarity
                if self._records:
                    for record in self._records:
                        # send each record to all clients
                        await self._runtime.publish_message(record, topic_id=self._topic)
                        agent_type, variant_config = self._agent_types[CONDUCTOR][0]
                        # TODO: FIX. THIS IS A HACK TO SEND TO UI. Wrap it in agentoutput first
                        trace = AgentTrace(agent_id=agent_type.type, agent_info=variant_config, outputs=record, session_id=self.session_id, inputs=AgentInput())
                        await self._runtime.publish_message(trace, topic_id=DefaultTopicId(type=MANAGER))

            # 3. Enter the main loop - now much simpler
            while True:
                try:
                    await asyncio.sleep(1)
                    # # Handle END signal
                    # if step.role == END:
                    #     logger.info("Host agent signaled flow completion")
                    #     break

                except ProcessingError as e:
                    # Non-fatal error - let the host agent decide how to recover
                    logger.error(f"Error in execution: {e}")
                    continue
                except (StopAsyncIteration, KeyboardInterrupt):
                    raise
                except FatalError:
                    raise
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}")
                    raise FatalError from e

        except (StopAsyncIteration, KeyboardInterrupt):
            logger.info("Flow terminated normally")
        except FatalError as e:
            logger.error(f"Fatal error: {e}")
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
        finally:
            await self._cleanup()

            # Log completion - use try/except to handle possible None cases
            try:
                call = weave.get_current_call()
                if call and hasattr(call, "ui_url"):
                    logger.info(f"Tracing link: 🍩 {call.ui_url}")
            except Exception:
                pass

    def _make_publish_callback(self) -> Callable[[FlowMessage], Awaitable[None]]:
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
