import asyncio
import json
from typing import Any, AsyncGenerator, Self

from autogen_core.models._types import UserMessage
from numpy import isin
import pydantic
import regex as re
from autogen_core import CancellationToken, FunctionCall, MessageContext
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool, ToolSchema
from promptflow.core._prompty_utils import parse_chat
from pydantic import Field, PrivateAttr

from buttermilk._core.agent import Agent, AgentInput, AgentOutput
from buttermilk._core.contract import (
    AllMessages,
    ConductorRequest,
    FlowMessage,
    GroupchatMessageTypes,
    ToolOutput,
    UserInstructions,
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.runner_types import Record
from buttermilk.bm import bm, logger
from buttermilk.utils._tools import create_tool_functions
from buttermilk.utils.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template,
    make_messages,
)


class LLMAgent(Agent):
    fail_on_unfilled_parameters: bool = Field(default=True)

    _tools_list: list[FunctionCall | Tool | ToolSchema | FunctionTool] = PrivateAttr(
        default_factory=list,
    )
    _json_parser: ChatParser = PrivateAttr(default_factory=ChatParser)
    _model_client: ChatCompletionClient = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def custom_agent_id(self) -> Self:
        # Set a custom name based on our major parameters
        components = self.id.split("-")
            
        components.extend([
            v
            for k, v in self.parameters.items()
            if k not in ["formatting", "description", "template"]
            and v and not re.search(r"\s", v)
        ])
        components = [c[:12] for c in components if c]
        self.id = "_".join(components)[:63]

        return self

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
    
    async def fill_template(
        self,
        inputs: AgentInput | None = None,
    ) -> list[Any]:
        """Fill the template with the given inputs and return a list of messages."""
        
        # 1. Prepare Jinja variables based on self.parameters mapping
        jinja_vars = {}
        for param_key in self.parameters.keys():
            # Check if the parameter value looks like a key for inputs.inputs
            if inputs and param_key in inputs.inputs and isinstance(inputs.inputs[param_key], str):
                jinja_vars[param_key] = inputs.inputs[param_key]
        
        # Render the template using Jinja2
        rendered_template, unfilled_vars = load_template(
            parameters=self.parameters,
            untrusted_inputs=jinja_vars,
        )
            
        # Prepare placeholder data separately
        record_messages = []
        context_messages = await self._model_context.get_messages()
        if inputs:
            record_messages = [rec.as_message() for rec in inputs.records if inputs]
        placeholders = {
                "records": record_messages,"context": context_messages,
            }
        
        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys. First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to divide into messages and set roles
        messages = make_messages(local_template=prompty, placeholders=placeholders)
              
        if (unfilled_vars := (set(unfilled_vars) - set(placeholders.keys()))):
            err = f"Template for agent {self.id} has unfilled parameters: {', '.join(unfilled_vars)}"
            if self.fail_on_unfilled_parameters:
                raise ProcessingError(err)
        return messages

    async def _execute_tools(
        self,
        calls: list[FunctionCall],
        cancellation_token: CancellationToken | None,
    ) -> list[ToolOutput]:
        """Execute the tools and return the results."""
        assert isinstance(calls, list) and all(
            isinstance(call, FunctionCall) for call in calls
        )

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._call_tool(call, cancellation_token) for call in calls],
        )
        results = [record for result in results for record in result if record is not None]

        return results

    async def _call_tool(
        self,
        call: FunctionCall,
        cancellation_token: CancellationToken | None,
    ) -> list[ToolOutput]:
        # Find the tool by name.
        tool = next((tool for tool in self._tools_list if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            results = await tool.run_json(arguments, cancellation_token)
        except Exception as e:
            return [ToolOutput(
                call_id=call.id,
                content=str(e),
                is_error=True,
                name=tool.name,
            )]
        if not isinstance(results, list):
            results = [results]
        outputs = []
        for result in results:
            result.call_id = call.id
            result.name = tool.name
            outputs.append(result)
        return outputs


    async def _create_agent_output(
        self,
        raw_content: str | list[Any],
        llm_metadata: dict | None = None,
        records: list[Record] | None = None,
        error_msg: str | None = None,
    ) -> AgentOutput:
        """Helper method to create AgentOutput instances."""
        outputs = {}
        content = ""
        if isinstance(raw_content, str):
            try:
                outputs = self._json_parser.parse(raw_content)
                # Pretty-print JSON if parsing is successful
                content = json.dumps(outputs, indent=2, sort_keys=True)
            except Exception as parse_error:
                logger.warning(f"Failed to parse LLM response as JSON: {parse_error}")
                content = raw_content # Use raw content if parsing fails
                if not error_msg: # Add parsing error if no other error exists
                    error_msg = f"JSON parsing error: {parse_error}"
        else:
            # Handle non-string content if necessary, or assume it's already structured
            outputs = raw_content # Or process as needed
            content = str(raw_content) # Basic string representation

        metadata = dict(self.parameters)
        if llm_metadata:
            metadata.update({
                k: v
                for k, v in llm_metadata.items()
                if v and k != "content" # Exclude content from metadata
            })

        return AgentOutput(
            source=self.id,
            role=self.role,
            outputs=outputs,
            content=content,
            metadata=metadata,
            records=records or [],
            error=error_msg,
        )

    async def listen(self, message: GroupchatMessageTypes, 
        ctx: MessageContext = None,
        **kwargs):
        """Save incoming messages for later use."""
        if message.content:
            # Map Buttermilk message types to LLM input types
            if isinstance(message, AgentOutput):
                await self._model_context.add_message(AssistantMessage(content=str(message.content), source=message.source))
            elif isinstance(message, UserInstructions):
                await self._model_context.add_message(UserMessage(content=str(message.content), source=message.source))
            else:
                # don't log other types of messages
                pass

    async def _process(
        self,
        message: FlowMessage,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput|None, None]:
        
        if isinstance(message, AgentInput):
            logger.debug(f"Agent {self.id} received {type(message)} directly in _process. Handling as standard input.")
        elif isinstance(message, GroupchatMessageTypes):
            logger.debug(f"Agent {self.id} received {type(message)} directly in _process. Ignoring.")
            return
        else:
            logger.warning(f"Agent {self.id} {self.role} dropping unknown message: {message.type} from {message.source}")
            return

        messages = await self.fill_template(inputs=message)
        records = message.records

        try:
            # Initial LLM call (potentially with tools)
            create_result = await self._model_client.create(
                messages=messages,
                tools=self._tools_list,
                cancellation_token=cancellation_token,
            )
            llm_metadata = create_result.model_dump(
                exclude_unset=True, exclude_none=True
            )

        except Exception as e:
            error_msg = f"Error during initial LLM call: {e}"
            logger.warning(error_msg)
            yield await self._create_agent_output(
                raw_content=error_msg,
                error_msg=error_msg,
                records=records,
            )
            return

        # --- Handle LLM Response ---
        if isinstance(create_result.content, str):
            # LLM returned a direct string response
            yield await self._create_agent_output(
                raw_content=create_result.content,
                llm_metadata=llm_metadata,
                records=records,
            )
        elif isinstance(create_result.content, list) and all(isinstance(item, FunctionCall) for item in create_result.content):
            # LLM returned tool calls (ensure it's a list of FunctionCall)
            try:
                tool_outputs = await self._execute_tools(
                    calls=create_result.content,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                error_msg = f"Error executing tools: {e}"
                logger.warning(error_msg, exc_info=True)
                yield await self._create_agent_output(
                    raw_content=error_msg,
                    error_msg=error_msg,
                    records=None,
                    llm_metadata=llm_metadata, # Include metadata from initial call
                )
                return

            # --- Reflection Phase (after tool execution) ---
            reflection_tasks = []
            for tool_result in tool_outputs:
                if tool_result.is_error:
                    # Optionally yield an error output for failed tool calls
                    error_msg = f"Tool call '{tool_result.name}' failed: {tool_result.content}"
                    logger.warning(error_msg)
                    yield await self._create_agent_output(
                        raw_content=error_msg,
                        error_msg=error_msg,
                        records=None,
                        # Consider adding tool call info to metadata here
                    )
                    continue # Skip reflection for failed tools? Or reflect on the error?
                
                if tool_result.send_to_ui:
                    # Ready to output as is
                    yield AgentOutput(
                        source=self.id,
                        role=self.role,
                        content=tool_result.content,
                        records=tool_result.results,
                    )
                    await asyncio.sleep(1) # let the message be absorbed
                
                try:
                    reflection_messages = messages.copy()
                    # Add the tool result message to the history for the reflection call
                    reflection_messages.extend(tool_result.messages)

                    # Create reflection task (call LLM without tools)
                    task = self._model_client.create(
                        messages=reflection_messages,
                        cancellation_token=cancellation_token,
                        # No tools passed for reflection call
                    )
                    reflection_tasks.append(task)
                except Exception as e:
                    error_msg = f"Error preparing reflection for tool '{tool_result.name}': {e}"
                    logger.warning(error_msg, exc_info=True)
                    yield await self._create_agent_output(
                        raw_content=error_msg, error_msg=error_msg, records=None
                    )
            # Process completed reflection tasks
            for task in asyncio.as_completed(reflection_tasks):
                try:
                    reflection_result = await task
                    reflection_metadata = reflection_result.model_dump(
                        exclude_unset=True, exclude_none=True
                    )
                    yield await self._create_agent_output(
                        raw_content=reflection_result.content,
                        llm_metadata=reflection_metadata,
                        records=None,
                    )
                except Exception as e:
                    error_msg = f"Error during reflection LLM call: {e}"
                    logger.warning(error_msg, exc_info=False)
                    yield await self._create_agent_output(
                        raw_content=error_msg, error_msg=error_msg, records=None
                )
        else:
             # Handle unexpected content type from LLM
             error_msg = f"Unexpected content type from LLM: {type(create_result.content)}"
             logger.error(error_msg)
             yield await self._create_agent_output(
                 raw_content=str(create_result.content), # Try string conversion
                 error_msg=error_msg,
                 records=records,
                 llm_metadata=llm_metadata,
             )
