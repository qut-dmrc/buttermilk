"""
Implements an Orchestrator using the autogen-core library for managing multi-agent interactions.

This module defines `AutogenOrchestrator`, which leverages Autogen's agent registration,
message routing (via topics and subscriptions), and runtime management to execute
complex workflows involving multiple LLM agents. It's designed to be configured via Hydra
and integrates with the Buttermilk agent and contract system.
"""

import asyncio
import itertools
from typing import Any, Mapping, Self  # Added type hints for clarity

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

    # Dynamically generates a unique topic ID for this specific orchestrator run.
    # Ensures messages within this run don't interfere with other concurrent runs.
    _topic: TopicId = PrivateAttr(
        default_factory=lambda: DefaultTopicId(type=f"{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:8]}")
    )

    async def _setup(self, request: RunRequest | None = None) -> None:
        """Initializes the Autogen runtime and registers all configured agents."""
        msg =f"Setting up AutogenOrchestrator for topic: {self._topic.type}"
        logger.info(msg)
        self._runtime = SingleThreadedAgentRuntime()

        # Register Buttermilk agents (wrapped in Adapters) with the Autogen runtime.
        await self._register_tools()
        await self._register_agents(params=request.model_dump())


        # Start the Autogen runtime's processing loop in the background.
        self._runtime.start()
        logger.debug("Autogen runtime started.")
        
        # Send a welcome message to the UI and start up the host agent
        await self._runtime.publish_message(ManagerMessage(content=msg), topic_id=MANAGER)
        await self._runtime.publish_message(StepRequest(role=CONDUCTOR, prompt = request.prompt), topic=CONDUCTOR)


    async def _register_agents(self, params: Mapping[str, Any]) -> None:
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
            for agent_cls, variant_config in step_config.get_configs(params=params):
                # Add extra params passed in at run-time where required
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
        Executes a step when the host agent delegates execution to the orchestrator.
        
        This method is simpler now as most of the flow control logic has moved to 
        the host agent, which now has the ability to execute steps directly.
        
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
        Delegates flow control to the CONDUCTOR (host) agent.
        
        This simplified version delegates the entire flow control logic to the host agent,
        which is now responsible for maintaining conversation state, tracking agent
        completions, and determining the next steps. The orchestrator simply serves
        as a message bus between agents.

        Returns:
            A StepRequest for the next step, or None if the conductor doesn't provide one.
        """
        logger.debug("Delegating flow control to host agent...")
        
        # Provide basic context about participants to the host
        conductor_inputs = {"participants": dict(self._agent_types.items())}
        
        # Create the request for the host agent
        request = ConductorRequest(
            inputs=conductor_inputs,
            prompt=self.parameters.get("task", ""),
            records=self._records,
        )

        # Send the message to the host agent 
        responses = await self._ask_agents(
            CONDUCTOR,
            message=request,
        )

        # Process the response
        valid_responses = [r for r in responses if isinstance(r, AgentOutput) and not r.is_error and isinstance(r.outputs, StepRequest)]

        if not valid_responses:
            logger.warning("Host agent did not return a valid StepRequest.")
            return None

        # Extract the next step
        next_step: StepRequest = valid_responses[0].outputs
        logger.debug(f"Host agent suggested next step for role: {next_step.role}")
        
        return next_step

    @weave.op
    async def _run(self, request: RunRequest | None = None) -> None:
        """
        Simplified main execution loop for the orchestrator.
        
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
            if request and (request.record_id or request.uri or request.prompt):
                fetch = FetchRecord(data=self.data)
                fetch_output = await fetch._run(
                    record_id=getattr(request, 'record_id', None),
                    uri=getattr(request, 'uri', None), 
                    prompt=getattr(request, 'prompt', None)
                )
                if fetch_output and fetch_output.results:
                    self._records = fetch_output.results
                    logger.debug(f"Loaded {len(self._records)} initial records")

            # 3. Enter the main loop - now much simpler
            while True:
                try:
                    # Get next step from host agent
                    step = await self._get_host_suggestion()
                    
                    # Handle no step case
                    if not step:
                        logger.debug("No step suggestion from host agent, waiting...")
                        await asyncio.sleep(5)
                        continue
                        
                    # Handle END signal
                    if step.role == END:
                        logger.info("Host agent signaled flow completion")
                        break
                        
                    # Get user confirmation if needed
                    if not await self._in_the_loop(step):
                        logger.debug("User rejected step, continuing to next suggestion")
                        continue
                        
                    # Execute the suggested step
                    await self._execute_step(step=step)
                    
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
                if call and hasattr(call, 'ui_url'):
                    logger.info(f"Tracing link: 🍩 {call.ui_url}")
            except Exception:
                pass
