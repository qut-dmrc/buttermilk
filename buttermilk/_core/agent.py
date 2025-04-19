from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Annotated, Any, AsyncGenerator, Callable, Self, Sequence, Union, TYPE_CHECKING

from autogen_core.model_context import UnboundedChatCompletionContext, ChatCompletionContext
import jmespath
import pydantic
from shortuuid import ShortUUID, uuid
import weave
from autogen_core import CancellationToken, MessageContext
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
)

from autogen_core.tools import BaseTool, FunctionTool
from buttermilk._core.log import logger
from buttermilk._core.config import DataSourceConfig, ToolConfig
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    AgentInput,
    AgentOutput,
    AllMessages,
    AssistantMessage,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    FlowMessage,
    GroupchatMessageTypes,
    ManagerResponse,
    OOBMessages,
    TaskProcessingComplete,  # Keep for type hints if needed elsewhere
    TaskProcessingStarted,  # Keep for type hints if needed elsewhere
    ToolOutput,
    UserInstructions,
    UserMessage,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.flow import KeyValueCollector
from buttermilk._core.types import Record
from buttermilk.utils.validators import convert_omegaconf_objects, lowercase_validator

# Forward reference for type hint
if TYPE_CHECKING:
    from buttermilk.bm import BM


#########
# Agent
#
# A simple class with a function that process input.
#
#
##########


class AgentConfig(BaseModel):
    id: str = Field(
        default_factory=uuid,
        description="A unique identifier for this agent.",
    )
    agent_obj: str = Field(
        default="",
        description="The object name to instantiate",
        exclude=True,
    )
    role: Annotated[str, AfterValidator(lowercase_validator)] = Field(
        ...,
        description="The role type that this agent fulfils.",
    )

    name: str = Field(description="The human friendly name of this agent type.")

    description: str = Field(
        ...,
        description="Short explanation of what this agent type does",
    )
    tools: list[ToolConfig] = Field(
        default=[],
        description="Tools the agent can invoke",
    )
    data: list[DataSourceConfig] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialisation parameters to pass to the agent",
    )
    # sequential_tasks removed - Orchestrator now handles sequence if needed
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of data to agent inputs",
    )
    outputs: dict[str, Any] = {}
    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }

    _validate_parameters = field_validator("parameters", mode="before")(convert_omegaconf_objects())


class Agent(AgentConfig):
    """Base Agent interface for all processing units.

    Agents are stateful. Context is stored internally by the agent and merged
    with data passed via AgentInput. _reset() clears state.
    """

    _trace_this = True  # Flag for potential future use, currently weave is explicit

    _records: list[Record] = PrivateAttr(default_factory=list)
    _model_context: ChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )
    _message_types_handled: type[FlowMessage] = PrivateAttr(default=AgentInput)  # TODO: Revisit if needed
    _heartbeat: asyncio.Queue = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))  # Keep for potential Autogen compatibility layer
    _data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)

    async def _check_heartbeat(self, timeout=60) -> bool:
        """Check if the heartbeat queue is empty."""
        try:
            await asyncio.wait_for(self._heartbeat.get(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _add_state_to_input(self, inputs: AgentInput) -> AgentInput:
        """Add local agent state to inputs before processing."""
        # Fill inputs based on input map defined in config
        inputs.inputs.update(self._data._resolve_mappings(self.inputs))

        # Add context and records from agent's memory
        inputs.context.extend(await self._model_context.get_messages())
        inputs.records.extend(self._records)

        return inputs

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken | None = None,
        source: str = "unknown",
        **kwargs,
    ) -> None:
        """Save incoming messages from *other* agents to update internal state."""
        # Look for matching roles in our inputs mapping
        if isinstance(message, (AgentOutput, ConductorResponse)):
            for key, mapping in self.inputs.items():
                if mapping and isinstance(mapping, str):
                    components = mapping.split(".", maxsplit=1)
                    if source.startswith(components[0]):
                        # Direct match
                        if len(components) == 1:
                            # No dot delineated field path: add the whole object
                            self._data.add(key, message.model_dump())
                        else:
                            # otherwise, try to find the value in the outputs dict using JMESPath
                            search_dict = {source: message.model_dump()}
                            if value := jmespath.search(mapping, search_dict):
                                self._data.add(key, value)

                        if key == "records":
                            # records are stored separately in our memory cache
                            if message.records:
                                self._records.extend(message.records)
                                continue
                else:
                    logger.warning(
                        f"Input mapping for {self.id} is too sophisticated and we haven't written the code to interpret it yet: {key} to {mapping}"
                    )
                    continue

        # Add message content to model context if appropriate
        if isinstance(message, (AgentOutput, ConductorResponse)):
            if message.content:
                await self._model_context.add_message(AssistantMessage(content=str(message.content), source=source))
        elif isinstance(message, (ToolOutput, UserInstructions)):
            # Don't add commands or empty user instructions to history
            if message.content and not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content), source=source))
        else:
            # don't log other types of messages to history
            pass

    @weave.op()
    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | ToolOutput | None:
        """Internal process function. Implement core agent logic here. Traced by Weave."""
        # Example:
        # logger.info(f"Agent {self.role} processing inputs: {inputs.inputs}")
        # await asyncio.sleep(1) # Simulate work
        # output_content = f"Processed: {inputs.prompt}"
        # return AgentOutput(role=self.role, content=output_content, inputs=inputs)
        raise NotImplementedError("Subclasses must implement the _process method.")

    async def _handle_control_message(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> OOBMessages | None:
        """Handle out-of-band control messages if needed."""
        logger.debug(f"Agent {self.role} {self.name} dropping control message: {message}")
        return None

    async def __call__(
        self,
        message: FlowMessage,
        cancellation_token: CancellationToken | None = None,
        source: str = "unknown",  # Identifier of the sender, if known
        **kwargs,
    ) -> AgentOutput | ToolOutput | OOBMessages | None:
        """Primary entry point called by the Orchestrator."""
        result = None
        call = None  # For weave logging

        try:
            if isinstance(message, AgentInput):
                # Prepare final input by adding agent state
                final_input = await self._add_state_to_input(message)

                # Execute core logic (traced via @weave.op on _process)
                _traced_process = weave.op(self._process, call_display_name=f"{self.name} {self.id}")
                result, call = await _traced_process.call(inputs=final_input, cancellation_token=cancellation_token, **kwargs)
                # Weave swallows errors. Raise here after tracing.
                if call.exception:
                    raise ProcessingError(f"Agent {self.id} hit error processing request: {call.exception}")

                # --- Evaluation Logic ---
                if isinstance(result, AgentOutput) and not result.is_error and final_input.records:
                    # Check for ground truth in records
                    ground_truth_record = next((r for r in final_input.records if getattr(r, "ground_truth", None) is not None), None)
                    # if ground_truth_record:
                    #     evaluation_score = await evaluate(
                    #         output=result,
                    #         ground_truth=ground_truth_record.ground_truth,
                    #         criteria=final_input.parameters.get("criteria"),  # Or get from self.params
                    #     )
                    #     if evaluation_score and call:
                    #         # Log evaluation to Weave trace associated with the agent's call
                    #         call.log({"evaluation": evaluation_score.model_dump()})
                # --- End Evaluation Logic ---

            elif isinstance(message, GroupchatMessageTypes):
                # Listen to messages from other agents to update state
                await self._listen(message=message, cancellation_token=cancellation_token, source=source, **kwargs)
                result = None  # Listen does not produce direct output

            elif isinstance(message, OOBMessages):
                # Handle control messages
                result = await self._handle_control_message(message=message, cancellation_token=cancellation_token, **kwargs)

            else:
                logger.warning(f"Agent {self.role} received unhandled message type: {type(message)}")
                result = None

        except Exception as e:
            logger.error(f"Error during agent {self.role} handle_message: {e}", exc_info=True)
            # Create an error output
            error_input = message if isinstance(message, AgentInput) else None
            result = AgentOutput(error=[str(e)], inputs=error_input)

        # Orchestrator is responsible for handling the result (e.g., routing AgentOutput)
        return result

    async def initialize(self, **kwargs) -> None:
        """Initialize the agent (e.g., load resources). Called by Orchestrator."""
        pass  # Default implementation

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state. Called by Orchestrator."""
        self._records = []
        self._model_context = UnboundedChatCompletionContext()
        self._data = KeyValueCollector()
        # Clear heartbeat queue if necessary
        while not self._heartbeat.empty():
            try:
                self._heartbeat.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"Agent {self.role} ({self.id}) reset.")
        pass  # Allow subclasses to add more reset logic
