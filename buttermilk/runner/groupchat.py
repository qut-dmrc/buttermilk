"""
Implements an Orchestrator using the autogen-core library for managing multi-agent interactions.

This module defines `AutogenOrchestrator`, which leverages Autogen's agent registration,
message routing (via topics and subscriptions), and runtime management to execute
complex workflows involving multiple LLM agents. It's designed to be configured via Hydra
and integrates with the Buttermilk agent and contract system.
"""

import asyncio
from typing import Any, Self  # Added type hints for clarity

import shortuuid
import weave  # weave is likely used for experiment tracking/logging.
from autogen_core import (
    AgentType,  # Represents a registered type of agent in the runtime.
    ClosureAgent,  # An agent defined by simple Python functions (closures).
    ClosureContext,  # Context provided to closure agent functions.
    DefaultTopicId,  # A standard implementation for topic identifiers.
    MessageContext,  # Context associated with a message during processing.
    SingleThreadedAgentRuntime,  # The core runtime managing agents and messages.
    TopicId,  # Abstract base class for topic identifiers.
    TypeSubscription,  # Defines a subscription based on message type and agent type.
    )
from pydantic import PrivateAttr, model_validator

# TODO: TaskProcessingComplete seems unused in this file. Consider removal.
# from buttermilk._core import TaskProcessingComplete
from buttermilk._core.agent import FatalError, ProcessingError  # Added Agent
from buttermilk._core.contract import (
    CONDUCTOR,
    # Added QualScore
    CONFIRM,
    END,
    MANAGER,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    ManagerMessage,
    ManagerRequest,  # Request sent to the MANAGER (UI/Human).
    ManagerResponse,
    RunRequest,  # Response received from the MANAGER.
    StepRequest,  # Defines a request for a specific step/agent execution.
    )

# TODO: Check if UserInstructions is actually used within this orchestrator's logic.
from buttermilk._core.orchestrator import Orchestrator  # Base class for orchestrators.
from buttermilk.agents.fetch import FetchRecord  # Agent for fetching data records.
from buttermilk.agents.ui.web import WebUIAgent
from buttermilk.bm import bm, logger  # Core Buttermilk instance and logger.
from buttermilk.libs.autogen import AutogenAgentAdapter  # Adapter to wrap Buttermilk agents for Autogen.


class AutogenOrchestrator(Orchestrator):
    """
    Orchestrates multi-agent workflows using the autogen-core library.

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
        _user_confirmation: Queue for receiving confirmation responses from the MANAGER.
        _topic: The main topic ID for this specific group chat instance, generated uniquely per run.
    """

    # Private attributes managed internally. Use PrivateAttr for Pydantic integration.
    _runtime: SingleThreadedAgentRuntime = PrivateAttr()
    _agent_types: dict[str, list[tuple[AgentType, Any]]] = PrivateAttr(default_factory=dict)
    _user_confirmation: asyncio.Queue[ManagerResponse] = PrivateAttr()

    # Dynamically generates a unique topic ID for this specific orchestrator run.
    # Ensures messages within this run don't interfere with other concurrent runs.
    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(type=f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}")
    )
    # TODO: Consider making the topic generation strategy more configurable or robust,
    # especially if persistence or specific naming conventions are needed.

    @model_validator(mode="after")
    def _initialize_internals(self) -> Self:
        """Initialize private attributes after standard Pydantic validation."""
        # Using a validator ensures these are set up after the main object initialization.
        self._user_confirmation = asyncio.Queue(maxsize=1)  # Queue for handling user/manager responses.
        # NOTE: _runtime and _agent_types are initialized within _setup.
        return self

    async def _setup(self, request: RunRequest | None = None) -> None:
        """Initializes the Autogen runtime and registers all configured agents."""
        msg =f"Setting up AutogenOrchestrator for topic: {self._topic.type}"
        logger.info(msg)
        self._runtime = SingleThreadedAgentRuntime()

        # Register Buttermilk agents (wrapped in Adapters) with the Autogen runtime.
        await self._register_tools()
        await self._register_agents()

        # Register a special agent to handle interactions with the MANAGER (UI/Human).
        await self._register_manager_interface(request.websocket, request.session_id)  # Renamed for clarity
        # Start the Autogen runtime's processing loop in the background.
        self._runtime.start()
        logger.debug("Autogen runtime started.")
        
        await self._send_ui_message(ManagerMessage(content=msg))


    async def _register_agents(self) -> None:
        """
        Registers Buttermilk agents (via Adapters) with the Autogen runtime.

        Iterates through the `self.agents` configuration, creating AutogenAgentAdapter
        instances for each agent variant and registering them with the runtime.
        Sets up subscriptions so agents listen on the main group chat topic and
        potentially role-specific topics.
        """
        logger.debug("Registering agents with Autogen runtime...")
        for role_name, step_config in self.agents.items():
            registered_for_role = []
            # `get_configs` likely yields tuples of (AgentClass, agent_variant_config)
            for agent_cls, variant_config in step_config.get_configs():
                # Define a factory function required by Autogen's registration.
                # This function creates an instance of the AutogenAgentAdapter,
                # wrapping the actual Buttermilk agent logic.
                # It captures loop variables (variant_config, agent_cls, self._topic.type)
                # to ensure the correct configuration is used when the factory is called.
                def adapter_factory(cfg=variant_config, cls=agent_cls, topic_type=self._topic.type):
                    # This log occurs when the factory is *called* by Autogen, not during registration.
                    # logger.debug(f"Instantiating adapter for agent {cfg.id} (Role: {cfg.role}, Class: {cls.__name__})")
                    return AutogenAgentAdapter(
                        agent_cfg=cfg,
                        agent_cls=cls,
                        topic_type=topic_type,  # Pass the main topic type
                    )

                # Register the adapter factory with the runtime.
                # `variant_config.id` should be a unique identifier for this specific agent instance/variant.
                agent_type: AgentType = await AutogenAgentAdapter.register(
                    runtime=self._runtime,
                    type=variant_config.id,  # Use the specific variant ID for registration
                    factory=adapter_factory,
                )
                logger.debug(f"Registered agent adapter: ID='{variant_config.id}', Role='{role_name}', Type='{agent_type}'")

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

    async def _register_tools(self) -> None:
        for role_name, step_config in self.tools.items():
            # instantiate known autogen native tools directly
            # TODO: this is a hack and should be changed
            from buttermilk.agents import SpyAgent

            if step_config.agent_obj == "SpyAgent":
                # Register the adapter factory with the runtime.
                # `variant_config.id` should be a unique identifier for this specific agent instance/variant.
                agent_type: AgentType = await SpyAgent.register(
                    runtime=self._runtime,
                    type=role_name.upper(),  # Use the specific variant ID for registration
                    factory=lambda: SpyAgent(save_dest=step_config.save_dest),
                )

                # Subscribe the newly registered agent type to the main group chat topic.
                # This allows it to receive general messages sent to the group.
                await self._runtime.add_subscription(
                    TypeSubscription(
                        topic_type=self._topic.type,  # Main group chat topic
                        agent_type=agent_type,
                    ),
                )
                logger.info(f"Registered spy {agent_type} and subscribed for topic '{self._topic.type}' ...")

    async def _register_manager_interface(self, websocket, session_id) -> None:
        """Registers an agent to communicate with the MANAGER (user)."""
        logger.debug(f"Registering manager interface agent '{MANAGER}'.")

        # Inject messages into the groupchat from the UI 
        agent_type = await WebUIAgent.register(
                    runtime=self._runtime,
                    type=MANAGER,
                    factory=lambda: WebUIAgent(websocket, session_id),
                )

        await self._runtime.add_subscription(
            TypeSubscription(
                topic_type=self._topic.type,  # Main group chat topic
                agent_type=agent_type,
            ),
        )
        return

        # This agent listens for ManagerResponse messages on relevant topics.
        async def handle_manager_response(
            _agent_ctx: ClosureContext,  # Context for the closure agent itself (unused here).
            message: ManagerResponse,  # The message received.
            _msg_ctx: MessageContext,  # Context of the message (unused here).
        ) -> None:
            """Puts received ManagerResponse messages into the confirmation queue."""
            # We only care about ManagerResponse messages here.
            # Other message types might arrive on the subscribed topics but will be ignored
            # due to the `unknown_type_policy="ignore"` setting below.
            if isinstance(message, ManagerResponse):
                logger.debug(f"Manager interface received confirmation: {message.dict()}")
                try:
                    # Place the confirmation message into the queue for the main loop to pick up.
                    self._user_confirmation.put_nowait(message)
                except asyncio.QueueFull:
                    # This shouldn't happen often with maxsize=1 if the main loop processes confirmations promptly.
                    logger.warning(f"User confirmation queue full. Discarding confirmation: {message.dict()}")
            else:
                # Log if an unexpected type arrives, although ignore policy should handle it.
                logger.warning(f"Manager interface received unexpected message type: {type(message)}")

        # Register the closure function as an agent named MANAGER.
        await ClosureAgent.register_closure(
            runtime=self._runtime,
            type=MANAGER,  # Agent ID/Name.
            closure=handle_manager_response,  # The async function to handle messages.
            subscriptions=lambda: [
                TypeSubscription(
                    topic_type=self._topic.type,  # Subscribe to the main group chat topic
                    agent_type=MANAGER, 
                )
            ],
            # If a message arrives that isn't ManagerResponse, just ignore it silently.
            unknown_type_policy="ignore",
        )
        logger.debug(f"Manager interface agent '{MANAGER}' registered and subscribed.")

    async def _ask_agents(
        self,
        role_name: str,  # Changed from step_name for clarity
        message: AgentInput | ConductorRequest | StepRequest,
    ) -> list[AgentOutput]:
        """
        Sends a message to all registered agents for a specific role and collects their outputs.

        Args:
            role_name: The UPPERCASE name of the role (e.g., "JUDGE", "SCORER").
            message: The input message (AgentInput, ConductorRequest, or StepRequest) to send.

        Returns:
            A list of AgentOutput objects received from the agents. Includes only successful responses.
        """
        tasks = []
        # Ensure role_name is uppercase to match the keys in _agent_types.
        role_key = role_name.upper()
        if role_key not in self._agent_types:
            logger.warning(f"Attempted to ask agents for unknown role: {role_key}")
            return []

        tasks = []
        # Copy the message to avoid potential modification issues if sent to multiple agents.
        input_message = message.model_copy(deep=True)  # Use deep copy for safety.
        logger.debug(f"Asking agents for role '{role_key}' with message type {type(input_message).__name__}")

        # Iterate through all registered agent types for the specified role.
        for agent_type, variant_config in self._agent_types[role_key]:
            try:
                # Get the specific agent instance ID from the runtime using its type.
                agent_id = await self._runtime.get(agent_type)  # This retrieves the instance ID created by the factory.
                logger.debug(f"Sending message to agent instance {agent_id} (Type: {agent_type}, Variant: {variant_config.id})")
                # Send the message asynchronously to the agent instance.
                # The runtime handles routing this to the adapter and then the Buttermilk agent.
                task = self._runtime.send_message(
                    message=input_message,
                    recipient=agent_id,
                )
                tasks.append(task)
            except Exception as e:
                # Log errors during message sending setup, e.g., if runtime can't find the agent.
                logger.error(f"Error preparing to send message to agent type {agent_type} for role {role_key}: {e}")

        # Wait for all the send_message tasks to complete and gather responses.
        # `return_exceptions=True` prevents one failed agent from stopping others.
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses: filter out errors/None and ensure they are AgentOutput.
        valid_outputs: list[AgentOutput] = []
        for i, response in enumerate(responses):
            agent_type, variant_config = self._agent_types[role_key][i]
            if isinstance(response, Exception):
                logger.error(f"Agent {variant_config.id} (Type: {agent_type}) failed processing message: {response}", exc_info=response)
            elif response is None:
                logger.warning(f"Agent {variant_config.id} (Type: {agent_type}) returned None.")
            elif isinstance(response, AgentOutput):
                if response.is_error:
                    logger.warning(f"Agent {variant_config.id} (Type: {agent_type}) returned an error output: {response.outputs}")
                else:
                    valid_outputs.append(response)
                    logger.debug(f"Agent {variant_config.id} (Type: {agent_type}) returned valid output.")
            else:
                # This case should ideally not happen if adapters work correctly.
                logger.error(f"Agent {variant_config.id} (Type: {agent_type}) returned unexpected type: {type(response)}. Output: {response}")

        logger.debug(f"Received {len(valid_outputs)} valid responses from role '{role_key}'.")
        return valid_outputs

    async def _send_ui_message(self, message: ManagerMessage | ManagerRequest) -> None:
        """
        Sends a message intended for the user interface (MANAGER).

        Args:
            message: The ManagerMessage or ManagerRequest to send.
        """
        # Publish to the specific MANAGER topic.
        manager_topic_id = DefaultTopicId(type=MANAGER)
        logger.debug(f"Publishing UI message ({type(message).__name__}) to topic '{MANAGER}'")
        await self._runtime.publish_message(message, topic_id=manager_topic_id)

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

    async def _execute_step(self, step: StepRequest, **kwargs) -> None:
        """
        Executes a single step in the workflow by sending a request to the appropriate agent(s).

        Args:
            step: A StepRequest object containing the role of the agent to execute
                  and the prompt/input for that agent.

        Raises:
            ProcessingError: If an error occurs during agent execution.
        """
        logger.debug(f"Executing step for role: {step.role}")
        try:
            # Prepare the input message for the agent(s).
            message = AgentInput(prompt=step.prompt, records=self._records)
            # Send the message to all agents registered for the specified role.
            # The result (list of AgentOutputs) is ignored here, assuming side effects or
            # subsequent steps rely on messages published by the agents.
            # TODO: Consider if the output from _ask_agents should be captured or processed here.
            _ = await self._ask_agents(
                step.role,
                message=message,
            )
            await asyncio.sleep(0.1)

        except Exception as e:
            # Wrap generic exceptions in ProcessingError for consistent handling.
            msg = f"Error during step execution for role '{step.role}': {e}"
            logger.error(msg)
            raise ProcessingError(msg) from e

    async def _get_host_suggestion(self) -> StepRequest | None:
        """
        Asks the CONDUCTOR agent to determine the next step in the workflow.

        Sends the current list of participants and the overall task prompt to the
        CONDUCTOR agent and expects a StepRequest in return, indicating the next
        agent role and prompt.

        Returns:
            A StepRequest for the next step, or None if the conductor doesn't provide one.
        """
        logger.debug("Asking CONDUCTOR for next step suggestion...")
        # Prepare the input for the CONDUCTOR agent.
        # Provide the list of currently registered agent types/roles.
        # TODO: `dict(self._agent_types.items())` might expose internal variant details.
        #       Consider passing just role names or a cleaner representation.
        conductor_inputs = {"participants": dict(self._agent_types.items())}
        # Include the overall task prompt and current records.
        request = ConductorRequest(
            inputs=conductor_inputs,
            prompt=self.parameters.get("task", ""),
            records=self._records,  # Get task description from flow parameters
        )

        # Ask the CONDUCTOR agent(s).
        responses = await self._ask_agents(
            CONDUCTOR,
            message=request,
        )

        # Determine the next step based on the response
        valid_responses = [r for r in responses if isinstance(r, AgentOutput) and not r.is_error and isinstance(r.outputs, StepRequest)]

        if not valid_responses:
            logger.warning("Conductor did not return a valid StepRequest.")
            return None

        # Use the first valid response
        # Add assertion for type checker clarity, although it might fail if conductor returns non-StepRequest.
        # TODO: Add more robust handling if the assertion fails or if valid_responses[0].outputs isn't StepRequest.
        assert isinstance(valid_responses[0].outputs, StepRequest), (
            f"Conductor returned unexpected type: Expected StepRequest, got {type(valid_responses[0].outputs)}"
        )
        next_step: StepRequest = valid_responses[0].outputs
        logger.debug(f"Conductor suggested next step for role: {next_step.role}")

        # TODO: The hardcoded sleep seems arbitrary. Consider removing or making configurable.
        #       Is it needed for timing issues or just pacing?
        await asyncio.sleep(5)
        return next_step

    @weave.op
    async def _run(self, request: RunRequest | None = None) -> None:
        """
        Main execution loop for the orchestrator.

        Sets up the Autogen runtime and agents, then enters a loop that repeatedly:
        1. Asks the CONDUCTOR agent for the next step.
        2. Optionally interacts with the MANAGER for confirmation (human-in-the-loop).
        3. Executes the suggested step by calling the appropriate agent(s).
        The loop continues until the CONDUCTOR signals the end (END role) or an
        unrecoverable error occurs.

        Args:
            request: An optional RunRequest containing initial data like record_id, uri, or prompt.
                     If provided, it fetches the initial record(s).
        """
        try:
            await self._setup(request)
            if request:
                fetch = FetchRecord(data=self.data)
                fetch_output = await fetch._run(record_id=request.record_id, uri=request.uri, prompt=request.prompt)
                # Fixed: Extract results list
                if fetch_output and fetch_output.results:
                    self._records = fetch_output.results

            while True:
                try:
                    # Loop until we receive an error
                    await asyncio.sleep(1)

                    # # Get next step in the flow
                    if not (step := await self._get_host_suggestion()):
                        # No next step at the moment; wait and try a bit
                        await asyncio.sleep(10)
                        continue

                    if step.role == END:
                        raise StopAsyncIteration("Host signaled that flow has been completed.")

                    if not await self._in_the_loop(step):
                        # User did not confirm plan; go back and get new instructions
                        continue

                    await self._execute_step(step=step)

                    # sleep a bit
                    await asyncio.sleep(5)

                except ProcessingError as e:
                    # non-fatal error
                    logger.error(f"Error in Orchestrator run: {e}")
                    continue
                except (StopAsyncIteration, KeyboardInterrupt):
                    raise
                except FatalError:
                    raise
                except Exception as e:  # This is only here for debugging for now.
                    logger.exception(f"Error in Orchestrator.run: {e}")
                    raise FatalError from e

        # Outer try/except for handling overall flow completion or fatal errors.
        except (StopAsyncIteration, KeyboardInterrupt):
            # Normal termination (END signal or user interruption).
            logger.debug(f"AutogenOrchestrator run loop for topic '{self._topic.type}' completed.")
        except FatalError as e:
            # Specific fatal error defined in Buttermilk.
            logger.error(f"Fatal error during orchestrator run: {e}")
        except Exception as e:
            # Catch any other unexpected exceptions during the main loop.
            logger.exception(f"Unexpected error during orchestrator run: {e}")
        finally:
            # Ensure cleanup happens regardless of how the loop exits.
            await self._cleanup()
