"""Provides a Buttermilk agent that wraps an Autogen AssistantAgent.

This module allows an Autogen `AssistantAgent` (which is capable of leveraging
LLMs, tools, and conversation history) to be used within the Buttermilk
framework by conforming to the `buttermilk._core.agent.Agent` interface.
"""

import json
from collections.abc import Sequence  # For type hinting sequences
from typing import Any, Self  # Standard typing utilities

import pydantic  # Pydantic core
from autogen_agentchat.agents import AssistantAgent  # The Autogen agent being wrapped
from autogen_agentchat.messages import (  # Autogen message types
    BaseChatMessage,  # Base for chat messages
)
from autogen_core import CancellationToken, MessageContext, RoutedAgent  # Autogen core components
from autogen_core.memory import Memory  # Autogen memory component
from autogen_core.models import (  # Autogen model client and message types
    AssistantMessage,
    ChatCompletionClient,
    UserMessage,
)
from autogen_core.tools import BaseTool  # Autogen base tool type
from pydantic import PrivateAttr  # For Pydantic private attributes

# Import the global Buttermilk instance getter
from buttermilk._core.bm_init import get_bm  # Changed from dmrc.get_bm to bm_init.get_bm based on file structure

bm = get_bm()  # Get the global Buttermilk instance

# Buttermilk core imports
from buttermilk._core.agent import Agent, AgentInput, AgentTrace  # Buttermilk base agent and message types
from buttermilk._core.contract import (  # Buttermilk contract types
    ConductorRequest,
    ErrorEvent,
    ToolOutput,  # Added ToolOutput
)
from buttermilk._core.log import logger  # Centralized logger
from buttermilk.utils._tools import create_tool_functions  # Utility for creating tool functions


class SimpleAutogenChatWrapper(RoutedAgent):
    """A simple example wrapper for an Autogen RoutedAgent.
    
    This class is primarily for illustrative purposes, showing a basic structure
    for how one might wrap an Autogen agent. It is not directly used by the
    main `AssistantAgentWrapper` but could serve as a starting point for simpler,
    custom Autogen agent integrations if needed.

    Note:
        This class appears to be incomplete or primarily illustrative, as the
        `_model_client` for the delegate is not properly initialized from constructor
        arguments in this example.

    Attributes:
        _model_client (ChatCompletionClient): Private attribute intended to hold
            the model client. Needs proper initialization.
        _delegate (AssistantAgent): The underlying Autogen AssistantAgent instance
            that this wrapper would delegate to.

    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initializes the SimpleAutogenChatWrapper.

        Args:
            name (str): The name of the agent.
            **kwargs: Additional keyword arguments for `RoutedAgent`.

        """
        super().__init__(name, **kwargs)
        # Note: _model_client needs to be properly initialized here,
        # perhaps from kwargs or a global/passed-in client.
        self._model_client: ChatCompletionClient = PrivateAttr()
        self._delegate = AssistantAgent(name, model_client=self._model_client)

    # Example message handler (decorator commented out by default)
    # @message_handler
    async def handle_messages(self, messages: Sequence[BaseChatMessage], ctx: MessageContext) -> None:
        """Handles incoming messages by streaming them to the delegate agent.

        Args:
            messages (Sequence[BaseChatMessage]): The sequence of messages to handle.
            ctx (MessageContext): The message context, including the cancellation token.

        """
        async for message in self._delegate.on_messages_stream(messages, ctx.cancellation_token):
            if message:
                # Process streamed messages from the delegate if necessary
                logger.debug(f"SimpleAutogenChatWrapper received message from delegate: {type(message)}")


class AssistantAgentWrapper(Agent):
    """Wraps an `autogen_agentchat.agents.AssistantAgent` to conform to the Buttermilk `Agent` interface.

    This agent acts as a bridge, allowing an Autogen AssistantAgent (which can
    interact with LLMs, use tools, and maintain conversation history) to be
    seamlessly integrated into a Buttermilk flow. It handles the translation
    of Buttermilk's `AgentInput` into a format Autogen expects and converts
    Autogen's responses back into Buttermilk's `AgentTrace` or `ToolOutput`.

    Key Configuration Parameters (expected in `AgentConfig.parameters`):
        - `model` (str): **Required**. The name of the LLM model to be used by the
          AssistantAgent (e.g., "gpt-4", "claude-3-sonnet"). This name must
          correspond to a configured LLM in `bm.llms`.
        - `system_prompt` (str): Optional. The system message/prompt to initialize
          the AssistantAgent with. Defaults to "You are a helpful assistant."

    Input Processing:
        - The agent primarily processes `AgentInput` messages from the Buttermilk flow.
        - It extracts the prompt and any relevant context from the `AgentInput`.
        - The `_process` method currently has a `NotImplementedError` placeholder
          for message translation logic. This needs to be completed to correctly
          format messages for `_assistant_agent.on_messages`.

    Output:
        - Aims to produce an `AgentTrace` containing the LLM's response or tool outputs.
        - Can return an `ErrorEvent` if processing fails.
    
    Private Attributes:
        _assistant_agent (AssistantAgent): The underlying Autogen AssistantAgent instance.
        _model_client (ChatCompletionClient): The Autogen chat completion client used by
            the `_assistant_agent`. Initialized from `bm.llms`.
        _tools_list (list[BaseTool[Any, Any]]): A list of Autogen-compatible tools
            derived from the agent's Buttermilk tool configurations.
        _memory (Sequence[Memory] | None): Optional Autogen memory component.
            (Currently not explicitly initialized or used in this wrapper beyond type hinting).
    """

    _assistant_agent: AssistantAgent = PrivateAttr()
    _model_client: ChatCompletionClient = PrivateAttr()
    _tools_list: list[BaseTool[Any, Any]] = PrivateAttr(default_factory=list)
    _memory: Sequence[Memory] | None = PrivateAttr(default=None)  # Not actively used in current impl

    @pydantic.model_validator(mode="after")
    def init_assistant_agent(self) -> Self:
        """Initializes the underlying Autogen `AssistantAgent` after Pydantic model validation.

        This method performs the following setup:
        1.  Retrieves the LLM model client (`_model_client`) from the global
            Buttermilk instance (`bm.llms`) based on the "model" parameter
            in `self.parameters`.
        2.  Initializes the list of tools (`_tools_list`) available to the
            Autogen agent using `create_tool_functions` from Buttermilk utilities,
            based on `self.tools` (from `AgentConfig`).
        3.  Determines the system message for the Autogen agent from the
            "system_prompt" parameter in `self.parameters` or uses a default.
        4.  Instantiates the `autogen_agentchat.agents.AssistantAgent` with the
            configured model client, tools, Buttermilk agent description, and system message.

        Returns:
            Self: The initialized `AssistantAgentWrapper` instance.

        Raises:
            ValueError: If the required "model" parameter is missing, if the specified
                        LLM model is not found in Buttermilk configurations, or if
                        `AssistantAgent` instantiation fails.

        """
        # 1. Initialize Model Client
        model_name = self.parameters.get("model")
        if not model_name or not isinstance(model_name, str):  # Ensure model_name is a string
            raise ValueError(f"Agent {self.role}: 'model' parameter (string) is required.")

        # Use the global bm instance to get the LLM client
        try:
            self._model_client = bm.llms.get_autogen_chat_client(model_name)
        except AttributeError as e:  # If model_name not found in bm.llms
            logger.error(f"Agent {self.role}: LLM model '{model_name}' not found in Buttermilk LLM configurations. Error: {e!s}")
            raise ValueError(f"LLM model '{model_name}' not configured for agent {self.role}.") from e

        # 2. Initialize Tools from self.tools (AgentConfig field)
        if self.tools:  # self.tools is inherited from AgentConfig
            try:
                # create_tool_functions should return List[BaseTool]
                self._tools_list = create_tool_functions(self.tools)
            except Exception as e:
                logger.error(f"Agent {self.role}: Failed to create tool functions: {e!s}")
                self._tools_list = []  # Default to empty list on error
        else:
            self._tools_list = []

        # 3. Determine System Message
        system_message_content = self.parameters.get("system_prompt", "You are a helpful assistant.")
        if not isinstance(system_message_content, str):  # Ensure system_prompt is a string
            logger.warning(f"Agent {self.role}: system_prompt is not a string. Using default.")
            system_message_content = "You are a helpful assistant."

        # 4. Instantiate AssistantAgent
        try:
            self._assistant_agent = AssistantAgent(
                name=self.role,  # Use Buttermilk role as Autogen agent name
                model_client=self._model_client,
                tools=self._tools_list,
                description=self.description,  # Use Buttermilk description
                system_message=system_message_content,
                # Other AssistantAgent parameters like llm_config, memory can be added here if needed
            )
        except Exception as e:
            logger.error(f"Failed to initialize autogen.AssistantAgent for role '{self.role}': {e!s}")
            raise ValueError(f"autogen.AssistantAgent initialization failed for role '{self.role}'") from e

        return self

    async def _process(
        self,
        *,
        message: AgentInput,  # Expects AgentInput from Buttermilk flow
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,  # Additional kwargs for flexibility
    ) -> AgentTrace | ToolOutput | ErrorEvent | None:  # Added ErrorEvent
        """Processes an incoming `AgentInput` message using the wrapped Autogen `AssistantAgent`.

        This method translates the Buttermilk `AgentInput` into a format suitable for
        Autogen's `AssistantAgent`, invokes its message processing, and then translates
        the response back into a Buttermilk `AgentTrace` or `ToolOutput`.

        The current implementation of message translation within this `_process` method
        is marked with a `NotImplementedError`. It needs to be completed to correctly
        format the `message.inputs` (especially a "prompt") and `message.context`
        into a list of `BaseChatMessage` objects for the Autogen agent.

        Args:
            message: The `AgentInput` message from the Buttermilk flow, containing
                the prompt, context, and other relevant data.
            cancellation_token: An optional `CancellationToken` for the operation.
            **kwargs: Additional keyword arguments (not currently used by this method).

        Returns:
            AgentTrace | ToolOutput | ErrorEvent | None:
            - `AgentTrace` containing the LLM's textual response or structured output.
            - `ToolOutput` if a tool call was made and results are directly processed here
              (though typically Autogen handles tool calls internally or returns
              `FunctionCall` objects in its response).
            - `ErrorEvent` if an error occurred during processing.
            - `None` if the message type is ignored or if processing is skipped
              (e.g., due to the `NotImplementedError` in message translation).
        
        Raises:
            NotImplementedError: Currently raised because the message translation logic
                from `AgentInput` to Autogen's `BaseChatMessage` list is not fully implemented.

        """
        # --- Agent Decision Logic ---
        # Decide whether to process this incoming message.
        # This wrapper primarily focuses on AgentInput for direct LLM interaction.
        # Other types like AgentTrace might be used to update context in a more complex setup.
        if not isinstance(message, AgentInput):
            logger.debug(f"AssistantAgentWrapper '{self.role}' received message of type {type(message)}, not AgentInput. Ignoring for direct processing.")
            # TODO: Future enhancement: Could use AgentTrace messages to update
            # the self._assistant_agent's memory/history if applicable.
            # if self._assistant_agent.memory and isinstance(message, AgentTrace):
            #    self._assistant_agent.memory.add_message(...)
            return None

        # Handle ConductorRequest specifically if needed (currently logs a warning)
        # The 'inputs' variable was not defined here, it should be 'message'.
        if isinstance(message, ConductorRequest):  # Corrected from 'inputs' to 'message'
            logger.warning(f"Agent '{self.role}' received ConductorRequest. Specific handling for ConductorRequest in AssistantAgentWrapper is not fully implemented.")
            # Depending on requirements, could extract data or return a specific response.
            # For now, let it fall through or return None if ConductorRequests shouldn't be processed by _assistant_agent.on_messages
            # return None

        if cancellation_token is None:
            cancellation_token = CancellationToken()

        # --- Message Translation: Buttermilk AgentInput to Autogen BaseChatMessage list ---
        messages_to_send: list[BaseChatMessage] = []

        # 1. Add prior context from AgentInput.context
        # This assumes message.context contains Autogen-compatible LLMMessages
        if message.context:
            for ctx_msg in message.context:
                if isinstance(ctx_msg, (UserMessage, AssistantMessage)):  # Check for Autogen types
                    messages_to_send.append(ctx_msg)
                else:
                    logger.warning(f"Skipping context message of unhandled type: {type(ctx_msg)} in AssistantAgentWrapper for '{self.role}'.")

        # 2. Add current input/prompt as a UserMessage
        # Determine content for the current UserMessage.
        # Priority: message.inputs["prompt"] > message.prompt > text from message.records
        current_input_content: str | None = None
        if message.inputs and "prompt" in message.inputs and message.inputs["prompt"]:
            current_input_content = str(message.inputs["prompt"])
        elif message.prompt:  # message.prompt comes from RunRequest via AgentInput
            current_input_content = message.prompt

        if not current_input_content and message.records:  # Fallback to record content
            # Simple concatenation of text from records. Might need more sophisticated handling.
            current_input_content = "\n".join(record.text for record in message.records if record.text).strip()

        if not current_input_content:
            # This is a critical point: if there's no content to send, the LLM call will likely fail or be meaningless.
            # The original code raised NotImplementedError here. Returning an ErrorEvent is more graceful.
            err_msg = f"AssistantAgentWrapper '{self.role}' received AgentInput with no usable content (prompt, inputs.prompt, or records text)."
            logger.error(err_msg)
            return ErrorEvent(source=self.agent_id, content=err_msg)
            # Original line:
            # raise NotImplementedError(f"AssistantWrapper {self.role} received message with no content/inputs. Skipping.")

        messages_to_send.append(UserMessage(content=current_input_content))

        # --- Call AssistantAgent's on_messages ---
        try:
            # `on_messages` is the primary method in Autogen's `RoutedAgent` (which AssistantAgent inherits from)
            # to process a sequence of messages.
            response_envelope = await self._assistant_agent.on_messages(
                messages=messages_to_send, cancellation_token=cancellation_token,
            )
        except Exception as e:
            err_msg = f"Agent '{self.role}' error during _assistant_agent.on_messages: {e!s}"
            logger.error(err_msg, exc_info=True)  # Add exc_info for stack trace
            return ErrorEvent(source=self.agent_id, content=err_msg)

        # --- Translate Autogen Response to Buttermilk AgentTrace ---
        output_content_str: str = ""
        output_data_dict: dict[str, Any] = {}
        processing_error_msg: str | None = None
        llm_metadata_dict: dict[str, Any] = {}

        if response_envelope and response_envelope.chat_message:
            autogen_chat_msg = response_envelope.chat_message

            if isinstance(autogen_chat_msg.content, str):
                output_content_str = autogen_chat_msg.content
                # Attempt to parse string content as JSON if it looks like structured output
                if output_content_str.strip().startswith("{") and output_content_str.strip().endswith("}"):
                    try:
                        parsed_json = json.loads(output_content_str)
                        if isinstance(parsed_json, dict):
                            output_data_dict = parsed_json  # Use parsed JSON as the primary output data
                        else:  # Parsed but not a dict, store raw text
                            output_data_dict = {"response_text": output_content_str}
                    except json.JSONDecodeError:
                        # Not valid JSON, store as raw text
                        output_data_dict = {"response_text": output_content_str}
                else:  # Not JSON-like, store as raw text
                    output_data_dict = {"response_text": output_content_str}

            elif autogen_chat_msg.content is not None:  # Handle non-string but non-None content (e.g., list of tool calls)
                # TODO: If autogen_chat_msg.content is a list of FunctionCall objects,
                # this needs to be handled to potentially create ToolOutput messages
                # or format them for AgentTrace. For now, stringifying.
                output_content_str = str(autogen_chat_msg.content)
                output_data_dict = {"raw_content": output_content_str, "content_type": type(autogen_chat_msg.content).__name__}
            # If content is None, output_content_str remains "" and output_data_dict remains {}

            # Extract LLM usage metadata if available (Autogen uses 'models_usage')
            if hasattr(autogen_chat_msg, "models_usage") and autogen_chat_msg.models_usage:
                usage_info = autogen_chat_msg.models_usage
                if hasattr(usage_info, "model_dump"):  # If it's a Pydantic model
                    llm_metadata_dict = usage_info.model_dump(exclude_unset=True)
                elif isinstance(usage_info, dict):  # If it's already a dict
                    llm_metadata_dict = usage_info
                else:  # Fallback for other types
                    try:
                        llm_metadata_dict = vars(usage_info)
                    except TypeError:
                        logger.warning(f"Could not convert models_usage of type {type(usage_info)} to dict for agent '{self.role}'.")
        else:
            processing_error_msg = "Autogen AssistantAgent returned no response or an empty chat_message."
            logger.warning(f"Agent {self.role}: {processing_error_msg}")
            # output_content_str will be empty, output_data_dict will be empty

        # Combine metadata from various sources
        final_metadata = {**(self.parameters or {}), **llm_metadata_dict}
        # Optionally, add inner messages from Autogen for debugging (can be verbose)
        # if hasattr(response_envelope, 'inner_messages') and response_envelope.inner_messages:
        #    final_metadata["_autogen_inner_messages"] = [msg.model_dump_json() for msg in response_envelope.inner_messages]

        # Construct and return AgentTrace
        # The 'call_id' for AgentTrace will be generated by AgentOutput's default factory.
        # If a specific call_id from Autogen (e.g., response_envelope.id) needs to be preserved,
        # it should be explicitly passed here to the AgentTrace constructor.
        return AgentTrace(
            agent_id=self.agent_id,  # From Buttermilk Agent
            agent_info=self._cfg,  # Buttermilk AgentConfig
            inputs=message,  # The original AgentInput that triggered this
            outputs=output_data_dict or output_content_str,  # Primary output; prefer dict if available
            metadata=final_metadata,
            error=[processing_error_msg] if processing_error_msg else [],
            # The `messages` field in AgentTrace is for the list of LLMMessages in the interaction.
            # This should include the messages sent to the LLM and the response received.
            messages=(messages_to_send + ([autogen_chat_msg] if response_envelope and response_envelope.chat_message else [])),  # type: ignore
        )

    async def initialize(self, **kwargs) -> None:
        """Initializes the agent. Called by `AutogenAgentAdapter` if used, or by Buttermilk's lifecycle.
        
        The core initialization of the underlying `AssistantAgent` (like setting up
        the LLM client, tools, and system message) is performed in the
        `init_assistant_agent` Pydantic model validator, which runs when the
        `AssistantAgentWrapper` instance is created.

        This `initialize` method primarily serves as a hook for any additional setup
        that might be required by an adapter or for specific Buttermilk agent
        lifecycle management beyond Pydantic model instantiation.

        Args:
            session_id (str): The session ID for this initialization context.
                Note that `self.session_id` (from the base `Agent` class) should
                already be set by the time this method is called if instantiated
                correctly by the Buttermilk framework.
            **kwargs: Additional keyword arguments that might be passed during
                initialization.

        """
        # Most of the specific AssistantAgent setup is in `init_assistant_agent` (Pydantic validator).
        # This method is part of the Buttermilk Agent lifecycle.
        # Call superclass initialize if it has any logic.
        await super().initialize(**kwargs)  # Pass kwargs to super if it accepts them
        logger.debug(f"Agent {self.role} (wrapper for AssistantAgent) initialized with session ID: {self.session_id}.")

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Resets the state of the underlying Autogen `AssistantAgent` and the wrapper.

        This method is called to clear any conversational history or internal
        state accumulated by both the wrapped Autogen agent and the Buttermilk
        wrapper, preparing it for a new sequence of interactions.

        Args:
            cancellation_token: An optional `CancellationToken` to signal if
                the reset operation should be aborted.

        """
        if hasattr(self, "_assistant_agent") and self._assistant_agent:
            await self._assistant_agent.on_reset(cancellation_token)
            logger.debug(f"Agent {self.role} (wrapper for AssistantAgent) has been reset.")
        else:
            # This might happen if on_reset is called before init_assistant_agent fully completes
            # or if init_assistant_agent failed.
            logger.warning(f"Agent {self.role}: _assistant_agent not available or not initialized at on_reset. Skipping Autogen agent reset.")

        # Call the superclass's on_reset to clear Buttermilk Agent state (_records, _model_context, _data)
        await super().on_reset(cancellation_token=cancellation_token)
        logger.debug(f"Buttermilk state for Agent {self.role} also reset.")
