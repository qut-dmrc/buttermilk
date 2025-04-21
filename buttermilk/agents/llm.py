import asyncio
from curses import meta
import json
import pprint
from types import NoneType
from typing import Any, AsyncGenerator, Callable, Optional, Self

from autogen_core.models._types import UserMessage
from psutil import Process
import pydantic
import regex as re
from autogen_core import CancellationToken, DefaultTopicId, FunctionCall, MessageContext, RoutedAgent, TopicId
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema
from promptflow.core._prompty_utils import parse_chat
from pydantic import BaseModel, Field, PrivateAttr
import weave

from buttermilk._core.agent import Agent, AgentInput, AgentOutput, ConductorResponse, ToolConfig, buttermilk_handler
from buttermilk._core.contract import (
    AllMessages,
    ConductorRequest,
    FlowMessage,
    GroupchatMessageTypes,
    LLMMessage,
    ToolOutput,
    UserInstructions,
    TaskProcessingComplete,
    ProceedToNextTaskSignal,
    OOBMessages,
)
from buttermilk._core.exceptions import ProcessingError

# Import only necessary classes from _core.llms
from buttermilk._core.llms import AutoGenWrapper, CreateResult
from buttermilk._core.types import Record

# Restore original bm import
from buttermilk.bm import bm, logger
from buttermilk.utils._tools import create_tool_functions
from buttermilk.utils.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
    make_messages,
)


class LLMAgent(Agent):
    """
    An LLM agent that uses a Buttermilk template/model configuration (via LLMAgent).
    It expects input conforming to AgentInput (handled by LLMAgent/Agent base)
    and aims to output results conforming to AgentReasons.
    """

    # @pydantic.model_validator(mode="after")
    # def custom_agent_id(self) -> Self:
    #     # Set a custom name based on our major variants
    #     components = self.role.split("-")

    #     components.extend(
    #         [v for k, v in self.variants.items() if k not in ["formatting", "description", "template"] and v and not re.search(r"\s", v)]
    #     )
    #     components = [c[:12] for c in components if c]
    #     self.role = "_".join(components)[:63]

    #     return self

    @pydantic.model_validator(mode="after")
    def init_model(self) -> Self:
        if self.parameters.get("model"):
            self._model_client = bm.llms.get_autogen_chat_client(
                self.parameters["model"],
            )
        else:
            raise ValueError("Must provide a model in the parameters.")

        return self

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        self._tools_list = create_tool_functions(self.tools)

        return self

    def __init__(
        self,
        # --- Arguments for LLMAgent/AgentConfig (handled by Pydantic) ---
        role: str,  # Default role for this agent type
        name: str,  # Friendly name (unique)
        description: str,  # Description for other agents
        parameters: dict[str, Any] = Field(default_factory=dict),
        tools: list[ToolConfig] = Field(default_factory=list),
        fail_on_unfilled_parameters: bool = Field(default=True),
        output_model: Optional[type[BaseModel]] = None,
        **kwargs,
    ):
        """
        Initializes the Judge agent.

        Args:
            role: The Buttermilk role.
            parameters: Configuration for the LLM, template, etc. (for LLMAgent).
            tools: List of tools the agent can use (Buttermilk format).
            fail_on_unfilled_parameters: LLMAgent setting.
            name: The friendly name for the agent.
            description: Description for the Autogen agent.
            system_message: System message for the Autogen agent.
            **kwargs: Additional arguments for AgentConfig (like 'id', 'inputs', 'outputs')
                      and potentially other RoutedAgent parameters.
        """
        # 1. Initialize Pydantic part (LLMAgent -> Agent -> AgentConfig)
        #    Pydantic handles collecting args matching its fields from kwargs.
        #    We pass all relevant args explicitly or via kwargs.
        #    `name` is also an AgentConfig field, so it's handled here too.
        Agent.__init__(
            self,
            role=role,
            name=name,  # Pass name to AgentConfig as well
            description=description,  # Pass description to AgentConfig
            parameters=parameters,
            tools=tools,
            **kwargs,  # Pass remaining AgentConfig fields (id, inputs, outputs etc.)
            # and potentially non-AgentConfig RoutedAgent fields too
        )
        self._fail_on_unfilled_parameters = fail_on_unfilled_parameters
        self._output_model: Optional[type[BaseModel]] = output_model
        self._tools_list: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(
            default_factory=list,
        )
        self._json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
        self._model_client: AutoGenWrapper = PrivateAttr()
        self._pause: bool = PrivateAttr(default=False)
        # 2. Initialize Autogen RoutedAgent part via the Mixin
        #    We need to explicitly call the superclass __init__ of the *mixin's* parent.
        #    Pass only the arguments relevant to RoutedAgent.
        #    Pydantic's __init__ already handled shared fields like 'name', 'description'.
        #    We only need to pass fields *specifically* for RoutedAgent if any beyond 'name'.
        #    Common RoutedAgent args: name, description, system_message, llm_client, etc.
        #    The mixin itself doesn't have an __init__, so we call RoutedAgent's directly.
        RoutedAgent.__init__(
            self,
            description=description,
        )
        self._topic_id: TopicId = DefaultTopicId(type=self.role)

    async def _fill_template(
        self,
        task_params: dict[str, Any],  # Accepts task-specific parameters
        inputs: dict[str, Any],
        context: list[LLMMessage] = [],
        records: list[Record] = [],
    ) -> list[Any]:
        """Fill the template with the given inputs and return a list of messages."""
        template = self.parameters.get("template", task_params.get("template", inputs.get("template")))
        if not template:
            raise ProcessingError("No template provided for agent {self.id}")

        # Render the template using Jinja2
        rendered_template, unfilled_vars = load_template(
            template=template,
            parameters=task_params,  # Use combined parameters for template source
            untrusted_inputs=inputs,  # Use derived jinja_vars for filling template
        )

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys. First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to divide into messages and set roles
        messages = make_messages(local_template=prompty, context=context, records=records)

        if unfilled_vars := (set(unfilled_vars) - set(["records", "context"])):
            err = f"Template for agent {self.role} has unfilled parameters: {', '.join(unfilled_vars)}"
            if self._fail_on_unfilled_parameters:
                raise ProcessingError(err)
        return messages

    def make_output(self, chat_result: CreateResult, inputs: AgentInput, schema: Optional[type[BaseModel]] = None) -> AgentOutput:
        """Helper to create AgentOutput from CreateResult."""
        output = AgentOutput()
        output.inputs = inputs.model_copy(deep=True)
        # Ensure exclude is a set or dict
        output.metadata = chat_result.model_dump(exclude={"content"})

        model_metadata = self.parameters
        model_metadata.update({"role": self.role, "agent_id": self.id, "name": self.name, "prompt": inputs.prompt})
        model_metadata.update(inputs.parameters)
        output.params = model_metadata

        # Handle schema validation first if applicable
        if schema and isinstance(chat_result.content, str):
            try:
                # model_validate_json returns an instance of the schema model
                output.outputs = schema.model_validate_json(chat_result.content)
                # Set content to a string representation
                output.content = output.outputs.model_dump_json()  # Use JSON string for content
                return output
            except Exception as e:
                error = f"Error parsing response from LLM: {e} into {schema.__name__}"
                logger.warning(error, exc_info=False)
                # Fall through to default JSON parsing or raw content handling
                # Add error to the output object?
                output.error.append(error)

        # Default handling: attempt JSON parsing or store raw content
        if isinstance(chat_result.content, str):
            try:
                # Store parsed dict in outputs
                output.outputs = self._json_parser.parse(chat_result.content)
                # Set content to formatted string version
                output.content = pprint.pformat(output.outputs)
            except Exception as parse_error:
                # If JSON parsing fails, store raw string content
                error = f"Failed to parse LLM response as JSON: {parse_error}. Storing raw content."
                logger.debug(error, exc_info=False)
                output.outputs = {"response": chat_result.content}
                output.content = chat_result.content
        elif chat_result.content is not None:  # Handle non-string, non-None content if necessary
            logger.warning(f"LLM response content is not a string: {type(chat_result.content)}. Storing as is.")
            output.outputs = {"response": chat_result.content}
            output.content = str(chat_result.content)
        # If content was None or parsing failed without fallback, outputs might be empty {}

        return output

    @weave.op()  # Add weave decorator to match base class and enable tracing
    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | ToolOutput | None:
        """Runs a single task or series of tasks."""
        try:
            messages = await self._fill_template(task_params=inputs.parameters, inputs=inputs.inputs, context=inputs.context, records=inputs.records)
        except ProcessingError as e:
            # We log here because weave swallows error results and we might lose it.
            logger.error(f"Unable to fill template for {self.id}: {e}")
            raise
        # call_chat can return CreateResult, list[ToolOutput], or None
        llm_result: CreateResult | list[ToolOutput] | None = await self._model_client.call_chat(
            messages=messages,
            tools_list=self._tools_list,
            cancellation_token=cancellation_token,
            reflect_on_tool_use=True,  # Assuming this handles tool calls internally now
            schema=self._output_model,
        )

        # Handle different return types
        if isinstance(llm_result, CreateResult):
            # Process normal LLM response
            agent_output = self.make_output(llm_result, inputs=inputs, schema=self._output_model)
            # Publish the result instead of returning
            await self.runtime.publish_message(agent_output, sender=self)
            return None  # Agents should not return directly in async Autogen flow
        elif isinstance(llm_result, list) and all(isinstance(item, ToolOutput) for item in llm_result):
            # If call_chat returns ToolOutput (e.g., for tool calls) publish each
            for tool_output in llm_result:
                await self.runtime.publish_message(tool_output, sender=self, topic_id=self.topic_id)
            return None  # Agents should not return directly
        elif llm_result is None:
            # Handle case where LLM call returns None (e.g., error, cancellation)
            logger.warning("LLM call returned None.")
            # Publish an error AgentOutput
            error_output = AgentOutput(error=["LLM call returned None"], inputs=inputs)
            await self.runtime.publish_message(error_output, sender=self)
            return None
        else:
            # Should not happen based on AutoGenWrapper signature, but good practice
            error_msg = f"Unexpected return type from call_chat: {type(llm_result)}"
            logger.error(error_msg)
            error_output = AgentOutput(error=[error_msg], inputs=inputs)
            if hasattr(self, "runtime") and self.runtime:
                await self.runtime.publish(error_output, sender=self)
            else:
                logger.error(f"Agent {self.id} has no runtime object to publish unexpected result error.")
            return None

    @buttermilk_handler(AgentInput)
    async def handle_agent_input(self, message: AgentInput) -> Optional[AgentOutput]:
        """
        Handles AgentInput messages from Autogen runtime.
        This is the primary entry point for LLM-based agents in Autogen group chats.
        """
        logger.info(f"LLMAgent '{self.name}' handling AgentInput: {message.inputs.get('prompt', 'No prompt')}")

        try:
            # Use the existing _process method
            result = await self._process(inputs=message)

            # _process now publishes the result via runtime.publish_message
            # but the Autogen handler also expects a return value
            if isinstance(result, AgentOutput):
                return result
            # If _process returned None or ToolOutput, we still need to return None
            # because the result was already published via runtime
            return None

        except Exception as e:
            logger.error(f"Error during LLMAgent handle_agent_input: {e}", exc_info=True)
            # Create and return an error AgentOutput
            return AgentOutput(error=[str(e)], inputs=message)

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        await super().on_reset(cancellation_token)
        self._current_task_index = 0
        self._last_input = None
        logger.debug(f"Agent {self.role} state reset.")
